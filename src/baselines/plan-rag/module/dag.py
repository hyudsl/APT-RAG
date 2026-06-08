from typing import List, Set, Optional, Dict, Any
import re


class DAGNode:
    def __init__(self, query: str):
        self.query = query.strip() 
        self.original_query = query.strip()
        self.meta: Dict[str, Any] = {}

        self.query_id = self._extract_query_id(self.query)
        if self.query_id == "UNKNOWN":
            self.meta["query_id_warning"] = "Failed to parse query id from reasoning plan."
            
        self.parent_nodes: Set['DAGNode'] = set()
        self.child_nodes: Set['DAGNode'] = set()

        self.depth: int = 0
        self.answer: Optional[str] = None
        self.search_query: Optional[str] = None
        self.search_results: Optional[List[Dict[str, Any]]] = None


    def update_query(self, query: str) -> None:
        self.query = query.strip()
        self.original_query = self.query
        self.query_id = self._extract_query_id(self.query)
        if self.query_id == "UNKNOWN":
            self.meta["query_id_warning"] = "Failed to parse query id from reasoning plan."


    def _extract_query_id(self, query: str) -> str:
        """Extract a query ID such as Q1.1 from a query string."""
        match = re.match(r'(Q\d*(?:\.\d+)*):?', query.strip())
        if match:
            return match.group(1)
        
        if query.strip().startswith("Q:"):
            return "Q"
            
        return "UNKNOWN"
    

    def add_parent(self, parent_node: 'DAGNode') -> None:
        """Add a parent node."""
        self.parent_nodes.add(parent_node)
        parent_node.child_nodes.add(self)


    def has_dynamic_tags(self) -> bool:
        """Return whether the query contains dynamic answer tags."""
        pattern = r'<A\d+\.\d+>'
        return bool(re.search(pattern, self.original_query))
    