from typing import List, Dict, Optional, Any


class TreeNode:
    def __init__(self, question:str, parent:Optional["TreeNode"]=None, retrieved: Optional[Any] = None, node_id: Optional[str] = None, **meta):
        self.question = question
        self.refined_question: Optional[str] = None
        self.query: Optional[str] = None
        self.answer: Optional[str] = None
        self.parent: Optional["TreeNode"] = parent
        self.children: List["TreeNode"] = []
        self.meta: Dict[str, Any] = meta
        self.retrieved:Optional[Any] = retrieved if retrieved is not None else {}
        self.memory: Dict[str, Any] = {}
        self.node_id: Optional[str] = node_id

    def add_child(self, question:str, node_id: Optional[str] = None, **meta):
        child = TreeNode(question, parent=self, node_id=node_id, **meta)
        self.children.append(child)
        return child
    
    def set_answer(self, answer:str, **meta):  
        self.answer = answer
        self.meta.update(meta)

    def set_memory(self, memory):
        self.memory = memory

    def to_dict(self):
        return {
            "question": self.question,
            "refined_question": self.refined_question,
            "query": self.query,
            "answer": self.answer,
            "meta": self.meta,
            "children": [child.to_dict() for child in self.children]
        }

    def to_retrieved_dict(self):
        return {
            "question": self.question,
            "refined_question": self.refined_question,
            "query": self.query,
            "retrieved": self.meta.get("retrieved", ""),
            "children": [child.to_retrieved_dict() for child in self.children]
        }