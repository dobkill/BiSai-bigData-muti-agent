"""Evidence workflow with explicit state transitions and bounded recovery.

Source extraction precedes this controller. It coordinates specialists, the
global assignment tool and fact verification, retaining immutable raw evidence.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import time

from .agents.event_units import EventUnitAgent
from .agents.one_to_one import OneToOneAlignmentAgent
from .agents.verification import VerificationAgent
from .agents.workflow_agents import CandidateRecallAgent, EntityResolutionAgent, ExceptionReviewAgent, SourceCompletenessAgent
from .relations import specific_platforms
from .utils import stable_id

logger = logging.getLogger(__name__)
METHOD = "evidence_state_workflow_v1"


@dataclass
class WorkflowState:
    stage: str = "extracted"
    round: int = 0
    transitions: list = field(default_factory=list)

    def enter(self, stage, reason="normal_flow", affected=None):
        self.transitions.append({"from": self.stage, "to": stage, "round": self.round,
                                 "reason": reason, "affected_news_ids": affected or []})
        self.stage = stage
        logger.info("工作流阶段：%s；第 %d 轮", stage, self.round)


class EvidenceWorkflow:
    def __init__(self, settings, llm):
        if not llm or not settings.alignment.llm_review or settings.alignment.mode != "one_to_one":
            raise ValueError("Agent 工作流需要启用模型、关联审查和 one_to_one 模式")
        self.settings, self.llm = settings, llm
        self.entity = EntityResolutionAgent(llm)
        self.source_review = SourceCompletenessAgent(llm)
        self.recall = CandidateRecallAgent(settings.alignment)
        self.reviewer = ExceptionReviewAgent(settings.workflow, settings.alignment, llm)
        self.verifier = VerificationAgent(settings.verification, settings.alignment, llm)
        self.audit = {}
        self.unassigned_observation_ids = []

    def verification_key(self, candidate, index, texts):
        """Reuse facts only while evidence, selected links and blockers agree.

        Assignment scores and unselected alternatives are retrieval context;
        they cannot invalidate an already checked, unchanged body of evidence.
        """
        selected_opinion = candidate.assignment.get("opinion_id")
        selected_links = [r for r in candidate.assignment.get("semantic_link_reviews", {}).get("opinions", [])
                          if r["opinion_id"] == selected_opinion]
        selected_links += [r.get("llm_review") for r in candidate.assignment.get("radar_assignments", [])]
        evidence = [{"observation": index[oid].model_dump(mode="json"), "text": texts.get(oid, "")}
                    for oid in candidate.observation_ids]
        return stable_id(json.dumps({"date": candidate.date, "id": candidate.id, "evidence": evidence,
                                     "selected_links": selected_links, "warnings": candidate.warnings,
                                     "verification_rules": self.settings.verification.model_dump(),
                                     "alignment_rules": self.settings.alignment.model_dump()},
                                    sort_keys=True, ensure_ascii=False))

    def run(self, observations, docs, store, *, on_round=None):
        # The pipeline reuses this controller across days. Evidence-search
        # diagnostics belong to this invocation, not to the previous day.
        self.reviewer.audit = []
        self.recall.routes = {}
        state, trace = WorkflowState(), []
        context = getattr(self.llm, "review_context", nullcontext)
        state.enter("source_completeness")
        started = time.perf_counter()
        with context():
            observations = self.source_review.run(observations, docs)
        trace.append({"agent": "source_completeness", "seconds": round(time.perf_counter()-started, 3), **self.source_review.audit})
        state.enter("entity_resolution")
        started = time.perf_counter()
        with context():
            observations = self.entity.run(observations, docs)
        observations = EventUnitAgent().run(docs, observations, keep_documents=True)
        # All later stages use the same resolved entities and activity scopes.
        store.replace_day(docs[0].date, observations, docs)
        observations = store.observations(docs[0].date)
        trace.append({"agent": "entity_resolution", "seconds": round(time.perf_counter()-started, 3), **self.entity.audit})
        state.enter("candidate_recall")
        overrides = self.recall.initial(observations, docs)
        trace.append({"agent": "candidate_recall", "indexed_observations": len(observations),
                      "raw_subject_proposals": len(overrides), "role": "retrieval_only"})
        index = {o.id: o for o in observations}
        news = sorted((o for o in observations if o.source == "open_source"), key=lambda o: (o.path, o.id))
        opinions = sorted((o for o in observations if o.source == "opinion"), key=lambda o: (o.path, o.id))
        radars = sorted((o for o in observations if o.source == "radar"), key=lambda o: (o.path, o.id))
        reviews, forced, inspections = {}, {}, {}
        recovery_history, verification_calls, link_calls = [], 0, 0
        round_summaries, link_fallbacks, verification_fallbacks = [], 0, 0
        verified_inputs, verification_reuses = {}, 0
        texts = {d.id: d.text for d in docs}
        total_alignment_seconds, total_verification_seconds = 0.0, 0.0
        while True:
            state.enter("activity_association")
            agent = OneToOneAlignmentAgent(self.settings.alignment, self.llm, recall_overrides=overrides,
                                           seed_reviews=reviews, extra_review_ids=forced)
            started = time.perf_counter()
            with context():
                candidates = agent.run(observations, docs)
            total_alignment_seconds += time.perf_counter()-started
            link_calls += agent.audit["llm_validated_reviews"]
            link_fallbacks += agent.audit["llm_review_fallbacks"]
            reviews = agent.annotations
            state.enter("global_assignment")
            state.enter("fact_verification")
            started = time.perf_counter()
            keys = {c.id: self.verification_key(c, index, texts) for c in candidates}
            pending = [c for c in candidates if keys[c.id] not in verified_inputs]
            with context():
                fresh_events = self.verifier.run(pending, observations, docs)
            fresh_index = {e.id: e for e in fresh_events}
            events = []
            for candidate in candidates:
                key = keys[candidate.id]
                if candidate.id in fresh_index:
                    event = fresh_index[candidate.id]
                    if event.semantic_review.get("status") == "validated":
                        verified_inputs[key] = (event.model_copy(deep=True), candidate.relation_status,
                            {k: deepcopy(candidate.assignment[k]) for k in ("semantic_association", "rule_relation_signals")
                             if k in candidate.assignment})
                else:
                    stored_event, status, annotations = verified_inputs[key]
                    event = stored_event.model_copy(deep=True)
                    candidate.relation_status = status
                    candidate.assignment.update(deepcopy(annotations))
                    verification_reuses += 1
                events.append(event)
            total_verification_seconds += time.perf_counter()-started
            verification_calls += self.verifier.audit["llm_verifications"]
            verification_fallbacks += self.verifier.audit["llm_verification_fallbacks"]
            round_summaries.append({"round": state.round,
                                    "new_verifications": len(pending), "reused_verifications": len(candidates)-len(pending),
                                    "paired_opinions": sum(c.assignment["opinion_id"] is not None for c in candidates),
                                    "classification": {s: sum(e.classification == s for e in events) for s in ("trusted", "untrusted")},
                                    "accepted": sum(e.accepted for e in events),
                                    "groups": [{"id": c.id, "observation_ids": c.observation_ids,
                                                "classification": next(e.classification for e in events if e.id == c.id)} for c in candidates]})
            if on_round:
                on_round({"round": state.round, "summary": round_summaries[-1],
                          "candidates": [c.model_dump(mode="json") for c in candidates],
                          "events": [e.model_dump(mode="json") for e in events],
                          "alignment_audit": agent.audit, "verification_audit": self.verifier.audit,
                          "unassigned_observation_ids": agent.unassigned_observation_ids})
            by_event = {e.id: e for e in events}
            targets = []
            for candidate in candidates:
                members = [index[oid] for oid in candidate.observation_ids]
                missing = {"opinion", "radar"} - {o.source for o in members}
                unresolved = any(r.startswith("semantic_association_not_confirmed") or r in {
                    "semantic_opinion_association_uncertain", "semantic_radar_association_uncertain",
                    "multiple_compatible_hypotheses"} for r in by_event[candidate.id].verification_reasons)
                if missing or unresolved:
                    targets.append((candidate, missing or {"opinion", "radar"}))
            if not targets:
                stop_reason = "no_recoverable_association_failure"
                break
            if state.round >= self.settings.workflow.max_recovery_rounds:
                stop_reason = "recovery_budget_exhausted"
                break
            state.enter("exception_review", "missing_source_or_unresolved_association",
                        [c.assignment["news_id"] for c, _ in targets])
            added = 0
            started = time.perf_counter()
            with context():
                for candidate, source_types in targets:
                    nid = candidate.assignment["news_id"]
                    n = index[nid]
                    i = next(i for i, item in enumerate(news) if item.id == nid)
                    for source in sorted(source_types):
                        pool = opinions if source == "opinion" else radars
                        reviewed = {opinions[k].id for key in reviews if len(key) == 2 and key[0] == i for k in [key[1]]} if source == "opinion" else {
                            radars[key[2]].id for key in reviews if len(key) == 3 and key[1] == i}
                        key = (nid, source)
                        excluded = reviewed | inspections.get(key, set())
                        proposed, inspected = self.reviewer.recover(n, pool, docs, self.recall, excluded)
                        inspections.setdefault(key, set()).update(inspected)
                        for oid in proposed:
                            overrides[(nid, oid)] = self.settings.alignment.minimum_pair_score
                            if source == "radar":
                                names = specific_platforms(n)
                                for name in names:
                                    if len(names) == 1 or name in specific_platforms(index[oid]):
                                        overrides[(nid + "|radar-slot|" + name, oid)] = self.settings.alignment.minimum_pair_score
                            forced.setdefault(nid, []).append(oid)
                            added += 1
                        recovery_history.append({"round": state.round+1, "news_id": nid, "source": source,
                                                 "inspected_ids": inspected, "proposed_ids": proposed})
            trace.append({"agent": "exception_review", "round": state.round+1,
                          "seconds": round(time.perf_counter()-started, 3), "proposed_candidates": added})
            if not added:
                stop_reason = "no_new_candidate_proposed"
                break
            state.round += 1
            state.enter("candidate_recall", "reviewer_requested_new_evidence", sorted(forced))
        state.enter("output", stop_reason)
        final_index = {c.id: c for c in candidates}
        for event in events:
            candidate = final_index[event.id]
            member_sources = {index[oid].source for oid in candidate.observation_ids}
            nid = candidate.assignment["news_id"]
            i = next(i for i, item in enumerate(news) if item.id == nid)
            coverage = {}
            for source, pool in (("opinion", opinions), ("radar", radars)):
                reviewed = {opinions[key[1]].id for key in reviews if len(key) == 2 and key[0] == i} if source == "opinion" else {
                    radars[key[2]].id for key in reviews if len(key) == 3 and key[1] == i}
                inspected = reviewed | inspections.get((nid, source), set())
                coverage[source] = {"corpus_size": len(pool), "inspected_ids": sorted(inspected),
                                    "complete": len(inspected) == len(pool)}
            event.semantic_review["workflow_diagnostics"] = {
                "source_coverage": "complete" if member_sources == {"open_source", "opinion", "radar"} else "missing_source",
                "association": event.semantic_review.get("response", {}).get("association", "uncertain"),
                "facts": "conflict" if any(v.status == "conflict" for v in event.verdicts.values()) else
                         "supported" if all(event.verdicts[f].status == "supported" for f in self.settings.verification.required_fields) else "insufficient",
                "search_coverage": coverage, "stop_reason": stop_reason}
            candidate.assignment["workflow"] = {"method": METHOD, "recovery": [r for r in recovery_history if r["news_id"] == nid],
                                                "recall_routes": {oid: self.recall.routes.get((nid, oid), []) for oid in candidate.observation_ids if oid != nid}}
        self.unassigned_observation_ids = agent.unassigned_observation_ids
        self.audit = {"method": METHOD, "rounds": state.round, "stop_reason": stop_reason,
                      "verification_reuses": verification_reuses,
                      "round_summaries": round_summaries,
                      "transitions": state.transitions, "recovery_reviews": self.reviewer.audit,
                      "recovery_proposals": sum(len(r["proposed_ids"]) for r in recovery_history),
                      "recovered_groups": sum(any(oid in c.observation_ids for r in recovery_history
                                                   if r["news_id"] == c.assignment["news_id"] for oid in r["proposed_ids"]) for c in candidates),
                      "paired_opinions": sum(c.assignment["opinion_id"] is not None for c in candidates)}
        trace += [{"agent": "alignment", "seconds": round(total_alignment_seconds, 3),
                   "rules": self.settings.alignment.model_dump(), **agent.audit, "llm_validated_reviews": link_calls,
                   "llm_review_fallbacks": link_fallbacks},
                  {"agent": "verification", "seconds": round(total_verification_seconds, 3),
                   "method": "three_source_semantic_verification_v4", "rules": self.settings.verification.model_dump(),
                   **self.verifier.audit, "llm_verifications": verification_calls,
                   "llm_verification_fallbacks": verification_fallbacks,
                   "classification": {s: sum(e.classification == s for e in events) for s in ("trusted", "untrusted")},
                   "accepted": sum(e.accepted for e in events)},
                  {"agent": "workflow", **self.audit}]
        return observations, candidates, events, trace
