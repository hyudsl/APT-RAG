from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from module.utils import Document, EvalMetrics


@dataclass
class TreeNode:
    """
    Tree of Questions node.
    """
    question: str
    level: int
    node_id: int
    
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    
    query: Optional[str] = None
    documents: List[Document] = field(default_factory=list)
    response: Optional[str] = None
    answer_span: Optional[str] = None
    
    eval_metrics: Optional[EvalMetrics] = None
    is_resolved: bool = False
    
    depends_on: Optional[List[int]] = None
    node_costs: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'TreeNode'):
        """Append a child node."""
        child.parent = self
        self.children.append(child)
    
    def get_ancestors(self) -> List['TreeNode']:
        """Return ancestors up to the root."""
        ancestors = []
        node = self.parent
        while node is not None:
            ancestors.append(node)
            node = node.parent
        return ancestors
    
    def get_context(self) -> str:
        """Return the accumulated ancestor QA context."""
        context_parts = []
        for ancestor in reversed(self.get_ancestors()):
            if ancestor.response:
                context_parts.append(f"Q: {ancestor.question}\nA: {ancestor.response}")
        return "\n\n".join(context_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a logging dictionary."""
        return {
            "node_id": self.node_id,
            "level": self.level,
            "question": self.question,
            "query": self.query,
            "response": self.response,
            "answer_span": self.answer_span,
            "is_resolved": self.is_resolved,
            "depends_on": self.depends_on,
            "node_costs": self.node_costs,
            "eval_metrics": self.eval_metrics.to_dict() if self.eval_metrics else None,
            "documents": [doc.to_dict() for doc in self.documents] if self.documents else [],
            "children_ids": [child.node_id for child in self.children]
        }

    def to_retrieved_dict(self) -> Dict[str, Any]: 
        """Convert the node to a recursive retrieval dictionary."""
        return {
            "node_id": self.node_id,
            "question": self.question,
            "query": self.query,
            "retrieved": [doc.to_dict() for doc in self.documents] if self.documents else [],
            "children": [child.to_retrieved_dict() for child in self.children]
        }


@dataclass
class TreeOfQuestionState:
    """Container for the full Tree of Questions state."""
    root: TreeNode
    current_nodes: List[TreeNode] = field(default_factory=list)
    all_nodes: List[TreeNode] = field(default_factory=list)
    total_queries: int = 0
    total_retrievals: int = 0
    
    def __post_init__(self):
        if not self.all_nodes:
            self.all_nodes = [self.root]
        if not self.current_nodes:
            self.current_nodes = [self.root]
    
    def add_node(self, node: TreeNode):
        self.all_nodes.append(node)
    
    def get_node_by_id(self, node_id: int) -> Optional[TreeNode]:
        for node in self.all_nodes:
            if node.node_id == node_id:
                return node
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full state to a dictionary."""
        return {
            "total_nodes": len(self.all_nodes),
            "total_queries": self.total_queries,
            "total_retrievals": self.total_retrievals,
            "nodes": {node.node_id: node.to_dict() for node in self.all_nodes}
        }