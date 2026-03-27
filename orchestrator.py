import json
import re
from dataclasses import dataclass

@dataclass
class Decision:
    node: str
    message: str

class DecisionMemory:
    def __init__(self):
        self.store = {}

    def build_key(self, intent, nodes):
        return hash(intent + str(len(nodes)))

    def get(self, key):
        return self.store.get(key)

    def save(self, key, value):
        self.store[key] = value

class FogOrchestrator:

    ALERT_WORDS = {"crash","emergency","critical","accident","brake"}

    def __init__(self, use_local_ai=True):
        self.use_ai = use_local_ai
        self.memory = DecisionMemory()

        if self.use_ai:
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(model="phi3", temperature=0.2)
            except:
                self.use_ai = False

    def decide(self, state, intent):

        if self._is_urgent(intent):
            return self._fast_path(state, "URGENT_REFLEX")

        key = self.memory.build_key(intent, state["fog_nodes"])
        cached = self.memory.get(key)
        if cached:
            return cached

        if self.use_ai:
            result = self._ai_plan(state, intent)
            self.memory.save(key, result)
            return result

        return self._fast_path(state, "FALLBACK")

    def _is_urgent(self, text):
        return any(word in text.lower() for word in self.ALERT_WORDS)

    def _fast_path(self, state, tag):
        node = sorted(state["fog_nodes"], key=lambda n: n["cpu"])[0]
        return {
            "target_node": node["id"],
            "reasoning": f"{tag}: Selected minimal load node"
        }

    def _ai_plan(self, state, intent):

        prompt = self._build_prompt(state, intent)

        try:
            raw = self.llm.invoke(prompt)
            parsed = self._extract_json(raw)
            if parsed:
                return parsed
        except:
            pass

        return self._fast_path(state, "AI_RECOVERY")

    def _build_prompt(self, state, intent):
        return f"""
Choose optimal fog node.

User Intent -> {intent}

System Snapshot -> {json.dumps(state)}

Return JSON:
{{"target_node":"id","reasoning":"short"}}
"""

    def _extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())
        return None