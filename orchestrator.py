import json
import re
from dataclasses import dataclass

# Machine Learning tools for our "Reflex" system
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@dataclass
class Decision:
    # A simple container to hold the final choice
    node: str
    message: str

class DecisionMemory:
    """This acts as the system's short-term memory (Cache).
    If we've seen a problem before, we instantly remember the answer instead of thinking again."""
    def __init__(self):
        self.memory_bank = {}

    def create_memory_key(self, intent, nodes):
        # Create a unique ID for this specific situation based on the request and available nodes
        return hash(intent + str(len(nodes)))

    def recall(self, key):
        return self.memory_bank.get(key)

    def memorize(self, key, value):
        self.memory_bank[key] = value

class FogOrchestrator:
    """This is the main brain of our vehicle system. It has two parts:
    1. A fast, instinctive reflex (Machine Learning Classifier)
    2. A slower, smarter cognitive brain (Generative AI / Phi-3)"""
    
    def __init__(self, use_local_ai=True):
        self.use_ai = use_local_ai
        self.memory = DecisionMemory()
        
        # Step 1: Train the "Reflexes" the moment the system starts up
        self._teach_reflexes_what_is_an_emergency()

        # Step 2: Wake up the "Smart Brain" (Phi-3 AI)
        if self.use_ai:
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(model="phi3", temperature=0.2)
            except Exception as error:
                print(f"Failed to wake up AI. Error: {error}")
                self.use_ai = False # Fall back to just reflexes

    def _teach_reflexes_what_is_an_emergency(self):
        """Trains an ultra-fast ML classifier to recognize danger without deep thinking."""
        
        # These are the examples we feed the model so it learns
        example_situations = [
            # --- EMERGENCIES (Label: 1) ---
            "crash ahead", "hit the brakes", "pedestrian on the road", 
            "tree fell down", "emergency stop", "avoid collision",
            
            # --- NORMAL REQUESTS (Label: 0) ---
            "optimize route", "save battery life", "play music", 
            "balance the traffic load", "dim the screen", "eco mode", "Prioritize Speed"
        ]
        
        # 1 means DANGER, 0 means NORMAL
        labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

        # The computer can't read words, so we convert sentences into a grid of numbers
        self.word_converter = CountVectorizer()
        sentences_as_numbers = self.word_converter.fit_transform(example_situations)

        # Train the lightning-fast Naive Bayes model to find the pattern
        self.reflex_model = MultinomialNB()
        self.reflex_model.fit(sentences_as_numbers, labels)

    def decide(self, vehicle_state, user_intent):
        """This is the main decision pipeline. It decides who should handle the problem."""
        
        # STEP 1: IS IT AN EMERGENCY?
        # If the ML reflex says it's dangerous, act instantly (< 1 millisecond)
        if self._is_emergency(user_intent):
            return self._trigger_fast_reflex(vehicle_state, "URGENT_REFLEX_ML")

        # STEP 2: HAVE WE SEEN THIS BEFORE?
        # Check the memory bank to save time
        memory_key = self.memory.create_memory_key(user_intent, vehicle_state["fog_nodes"])
        remembered_decision = self.memory.recall(memory_key)
        
        if remembered_decision:
            return remembered_decision

        # STEP 3: THINK DEEPLY (Generative AI)
        # If it's safe and new, let the Phi-3 AI analyze the best route
        if self.use_ai:
            ai_decision = self._ask_smart_ai(vehicle_state, user_intent)
            self.memory.memorize(memory_key, ai_decision) # Save for next time
            return ai_decision

        # STEP 4: BACKUP PLAN
        # If all else fails, do the safest thing possible
        return self._trigger_fast_reflex(vehicle_state, "FALLBACK")

    def _is_emergency(self, text):
        """Asks the trained ML model if the text sounds dangerous."""
        words_as_numbers = self.word_converter.transform([text])
        prediction = self.reflex_model.predict(words_as_numbers)[0]
        
        # Returns True if the model predicted 1 (DANGER)
        return prediction == 1

    def _trigger_fast_reflex(self, vehicle_state, reason_tag):
        """Instantly finds the node with the lowest CPU load to prevent crashes."""
        best_node = sorted(vehicle_state["fog_nodes"], key=lambda node: node["cpu"])[0]
        
        return {
            "target_node": best_node["id"],
            "reasoning": f"{reason_tag}: ML classified as safety-critical. Bypassed LLM."
        }

    def _ask_smart_ai(self, vehicle_state, intent):
        """Packages the problem and sends it to the Phi-3 LLM to solve."""
        prompt = self._create_prompt_for_ai(vehicle_state, intent)
        
        try:
            raw_answer = self.llm.invoke(prompt)
            structured_answer = self._extract_json_from_text(raw_answer)
            
            if structured_answer:
                return structured_answer
        except:
            pass # If the AI crashes or times out, we catch it silently
            
        # If the AI failed, trigger the safety backup
        return self._trigger_fast_reflex(vehicle_state, "AI_RECOVERY")

    def _create_prompt_for_ai(self, vehicle_state, intent):
        """Formats the data so the AI knows exactly what we want."""
        return f"""
Choose optimal fog node.
User Intent -> {intent}
System Snapshot -> {json.dumps(vehicle_state)}
Return JSON:
{{"target_node":"id","reasoning":"short"}}
"""

    def _extract_json_from_text(self, text):
        """Hunts for the JSON bracket inside the AI's raw text output."""
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())
        return None