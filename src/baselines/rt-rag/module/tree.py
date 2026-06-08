


#---------------------------------------------------- Binary Tree Structure -----------------------------------------------
global_node_counter = 0
class QuestionNode:
    def __init__(self, question, q_type="None", subq1=None, subq2=None, parent=None, is_left_child=True, question_id=None):
        global global_node_counter
        if question_id is None:
            self.id = f"N{global_node_counter}"
            global_node_counter += 1
        else:
            self.id = question_id
        
        self.question = question
        self.display_question = question
        self.type = q_type
        self.left = None
        self.right = None
        self.parent = parent
        self.is_left_child = is_left_child
        self.depends_on = None
        self.subq1_text = subq1
        self.subq2_text = subq2
        self.answer = None  # Store the node's answer
        self.meta = {}  # Provenance/logging: retrieved, context, etc.

    def to_dict(self):
        """Serialize tree for logging (Monaco-style)."""
        return {
            "question": self.question,
            "display_question": getattr(self, "display_question", self.question),
            "answer": self.answer,
            "type": self.type,
            "meta": self.meta,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    def __str__(self):
        return f"ID: {self.id}, Type: {self.type}, Q: {self.display_question[:50]}{'...' if len(self.display_question) > 50 else ''}"
