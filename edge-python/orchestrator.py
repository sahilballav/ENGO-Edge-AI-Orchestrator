import json
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@dataclass
class Decision:
    node: str
    message: str

class DecisionMemory:
    def __init__(self):
        self.memory_bank = {}

    def create_memory_key(self, intent, nodes):
        return hash(intent + str(len(nodes)))

    def recall(self, key):
        return self.memory_bank.get(key)

    def memorize(self, key, value):
        self.memory_bank[key] = value

class FogOrchestrator:
    def __init__(self, use_local_ai=True):
        self.use_ai = use_local_ai
        self.memory = DecisionMemory()
        
        self._teach_reflexes_what_is_an_emergency()

        if self.use_ai:
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(model="phi3", temperature=0.2)
            except Exception as error:
                print(f"Failed to wake up AI. Error: {error}")
                self.use_ai = False 

    def _teach_reflexes_what_is_an_emergency(self):
        example_situations = [
            # --- EMERGENCIES (Label: 1) ---
            "crash ahead", "hit the brakes", "pedestrian on the road", 
            "tree fell down", "emergency stop", "avoid collision", 
            "THERMAL OVERLOAD DETECTED", # <-- Added our new camera trigger!
            
            # --- NORMAL REQUESTS (Label: 0) ---
            "optimize route", "save battery life", "play music", 
            "balance the traffic load", "dim the screen", "eco mode", "Prioritize Speed"
        ]
        
        labels = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] # Updated labels to match

        self.word_converter = CountVectorizer()
        sentences_as_numbers = self.word_converter.fit_transform(example_situations)

        self.reflex_model = MultinomialNB()
        self.reflex_model.fit(sentences_as_numbers, labels)

    def decide(self, vehicle_state, user_intent):
        if self._is_emergency(user_intent):
            return self._trigger_fast_reflex(vehicle_state, "URGENT_REFLEX_ML")

        memory_key = self.memory.create_memory_key(user_intent, vehicle_state["fog_nodes"])
        remembered_decision = self.memory.recall(memory_key)
        
        if remembered_decision:
            return remembered_decision

        if self.use_ai:
            ai_decision = self._ask_smart_ai(vehicle_state, user_intent)
            self.memory.memorize(memory_key, ai_decision)
            return ai_decision

        return self._trigger_fast_reflex(vehicle_state, "FALLBACK")

    def _is_emergency(self, text):
        words_as_numbers = self.word_converter.transform([text])
        prediction = self.reflex_model.predict(words_as_numbers)[0]
        return prediction == 1

    def _trigger_fast_reflex(self, vehicle_state, reason_tag):
        best_node = sorted(vehicle_state["fog_nodes"], key=lambda node: node["cpu"])[0]
        return {
            "target_node": best_node["id"],
            "reasoning": f"{reason_tag}: ML classified as safety-critical. Bypassed LLM."
        }

    def _ask_smart_ai(self, vehicle_state, intent):
        prompt = self._create_prompt_for_ai(vehicle_state, intent)
        try:
            raw_answer = self.llm.invoke(prompt)
            structured_answer = self._extract_json_from_text(raw_answer)
            if structured_answer:
                return structured_answer
        except:
            pass 
            
        return self._trigger_fast_reflex(vehicle_state, "AI_RECOVERY")

    def _create_prompt_for_ai(self, vehicle_state, intent):
        return f"""
Choose optimal fog node.
User Intent -> {intent}
System Snapshot -> {json.dumps(vehicle_state)}
Return JSON:
{{"target_node":"id","reasoning":"short"}}
"""

    def _extract_json_from_text(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())
        return None