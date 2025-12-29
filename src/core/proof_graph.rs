//! Proof Graph Module
//!
//! Formalizes reasoning as a directed acyclic graph (DAG) where:
//! - Nodes = propositions, beliefs, or intermediate states
//! - Edges = inference steps with justifications
//! - Paths = complete reasoning chains from problem to solution
//!
//! Mathematical Foundation:
//! G = (V, E) where:
//! - V = {v₁, v₂, ..., vₙ} are proposition nodes
//! - E ⊆ V × V are justified inference edges
//! - Each edge e = (u, v) has:
//!   * Inference rule: r ∈ R
//!   * Confidence: c ∈ [0, 1]
//!   * Justification: J (proof, evidence, or derivation)
//!
//! Key Properties:
//! 1. Acyclic: No circular reasoning
//! 2. Traceable: Every conclusion has a path to axioms/premises
//! 3. Verifiable: Each edge can be independently checked
//! 4. Composable: Subgraphs can be reused

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::core::ThoughtState;

// ============================================================================
// PART 1: GRAPH STRUCTURE
// ============================================================================

/// Proof graph: G = (V, E)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalProofGraph {
    /// Nodes: propositions or states
    pub nodes: HashMap<NodeId, FormalProofNode>,
    /// Edges: inference steps
    pub edges: Vec<FormalProofEdge>,
    /// Root nodes (axioms, premises, given facts)
    pub roots: HashSet<NodeId>,
    /// Goal nodes (conclusions to prove)
    pub goals: HashSet<NodeId>,
    /// Metadata
    pub metadata: GraphMetadata,
}

pub type NodeId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub id: String,
    pub created_at: u64,
    pub problem_description: String,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub max_depth: usize,
}

// ============================================================================
// PART 2: NODES (Propositions)
// ============================================================================

/// Node in proof graph: represents a proposition or state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalProofNode {
    /// Unique identifier
    pub id: NodeId,
    /// Type of node
    pub node_type: FormalNodeType,
    /// Content (proposition, state, or claim)
    pub content: NodeContent,
    /// Confidence in this node (0-1)
    pub confidence: f64,
    /// Support: how many edges lead TO this node
    pub in_degree: usize,
    /// Consequences: how many edges lead FROM this node
    pub out_degree: usize,
    /// Depth from root (0 = axiom/premise)
    pub depth: usize,
    /// Whether this node has been verified
    pub verified: bool,
    /// Verification method used
    pub verification_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormalNodeType {
    /// Axiom or given premise (root)
    Axiom,
    /// Intermediate derived proposition
    Derived,
    /// Goal to prove
    Goal,
    /// Assumption (may be discharged later)
    Assumption,
    /// Contradiction (for proof by contradiction)
    Contradiction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeContent {
    /// Text proposition
    Proposition(String),
    /// Thought state vector
    State(ThoughtState),
    /// Logical formula (simplified)
    Formula(String),
    /// Numerical value
    Value(f64),
    /// Composite (multiple sub-propositions)
    Composite(Vec<String>),
}

impl FormalProofNode {
    pub fn new_axiom(content: NodeContent, confidence: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            node_type: FormalNodeType::Axiom,
            content,
            confidence,
            in_degree: 0,
            out_degree: 0,
            depth: 0,
            verified: true, // Axioms are assumed true
            verification_method: Some("axiom".to_string()),
        }
    }
    
    pub fn new_derived(content: NodeContent, confidence: f64, depth: usize) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            node_type: FormalNodeType::Derived,
            content,
            confidence,
            in_degree: 0,
            out_degree: 0,
            depth,
            verified: false,
            verification_method: None,
        }
    }
    
    pub fn new_goal(content: NodeContent) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            node_type: FormalNodeType::Goal,
            content,
            confidence: 0.0, // To be determined
            in_degree: 0,
            out_degree: 0,
            depth: 0, // Will be calculated
            verified: false,
            verification_method: None,
        }
    }
}

// ============================================================================
// PART 3: EDGES (Inference Steps)
// ============================================================================

/// Edge in proof graph: represents an inference step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalProofEdge {
    /// Unique identifier
    pub id: String,
    /// Source node (premise)
    pub from: NodeId,
    /// Target node (conclusion)
    pub to: NodeId,
    /// Inference rule applied
    pub rule: FormalInferenceRule,
    /// Confidence in this inference (0-1)
    pub confidence: f64,
    /// Justification for this step
    pub justification: Justification,
    /// Whether this edge has been verified
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormalInferenceRule {
    /// Modus Ponens: A, A→B ⊢ B
    ModusPonens,
    /// Modus Tollens: ¬B, A→B ⊢ ¬A
    ModusTollens,
    /// Conjunction: A, B ⊢ A∧B
    Conjunction,
    /// Disjunction: A ⊢ A∨B
    Disjunction,
    /// Simplification: A∧B ⊢ A
    Simplification,
    /// Transitivity: A→B, B→C ⊢ A→C
    Transitivity,
    /// Resolution: A∨B, ¬B∨C ⊢ A∨C
    Resolution,
    /// Substitution: Replace equals with equals
    Substitution,
    /// Generalization: Specific to general
    Generalization,
    /// Specialization: General to specific
    Specialization,
    /// Analogy: Similar cases imply similar conclusions
    Analogy,
    /// Induction: Pattern from examples
    Induction,
    /// Abduction: Best explanation
    Abduction,
    /// Custom rule with name
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Justification {
    /// Logical derivation
    Logical { steps: Vec<String> },
    /// Mathematical proof
    Mathematical { proof: String },
    /// Empirical evidence
    Empirical { evidence: Vec<String>, confidence: f64 },
    /// Definitional (true by definition)
    Definitional { definition: String },
    /// Assumption (may be discharged)
    Assumption { reason: String },
    /// Reference to external knowledge
    Reference { source: String },
    /// Operator transformation
    OperatorTransform { operator_id: String, energy: f64 },
}

impl FormalProofEdge {
    pub fn new(
        from: NodeId,
        to: NodeId,
        rule: FormalInferenceRule,
        confidence: f64,
        justification: Justification,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from,
            to,
            rule,
            confidence,
            justification,
            verified: false,
        }
    }
}

// ============================================================================
// PART 4: PATHS (Reasoning Chains)
// ============================================================================

/// A path from premise to conclusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofPath {
    /// Sequence of nodes
    pub nodes: Vec<NodeId>,
    /// Sequence of edges
    pub edges: Vec<String>, // Edge IDs
    /// Overall confidence (minimum along path)
    pub confidence: f64,
    /// Path length
    pub length: usize,
    /// Whether this path is valid
    pub valid: bool,
    /// Verification result
    pub verification: Option<PathVerification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathVerification {
    /// Whether path is logically sound
    pub sound: bool,
    /// Whether path is complete (no gaps)
    pub complete: bool,
    /// Weakest link (lowest confidence edge)
    pub weakest_link: Option<String>,
    /// Issues found
    pub issues: Vec<String>,
}

impl ProofPath {
    /// Calculate path confidence (minimum edge confidence)
    pub fn calculate_confidence(&mut self, graph: &FormalProofGraph) {
        let mut min_confidence = 1.0;
        
        for edge_id in &self.edges {
            if let Some(edge) = graph.edges.iter().find(|e| &e.id == edge_id) {
                if edge.confidence < min_confidence {
                    min_confidence = edge.confidence;
                }
            }
        }
        
        self.confidence = min_confidence;
    }
    
    /// Verify path validity
    pub fn verify(&mut self, graph: &FormalProofGraph) -> PathVerification {
        let mut sound = true;
        let mut complete = true;
        let mut weakest_link = None;
        let mut issues = Vec::new();
        let mut min_confidence = 1.0;
        
        // Check each edge
        for edge_id in &self.edges {
            if let Some(edge) = graph.edges.iter().find(|e| &e.id == edge_id) {
                if !edge.verified {
                    sound = false;
                    issues.push(format!("Unverified edge: {}", edge_id));
                }
                
                if edge.confidence < min_confidence {
                    min_confidence = edge.confidence;
                    weakest_link = Some(edge_id.clone());
                }
            } else {
                complete = false;
                issues.push(format!("Missing edge: {}", edge_id));
            }
        }
        
        // Check node connectivity
        for i in 0..self.nodes.len().saturating_sub(1) {
            let from = &self.nodes[i];
            let to = &self.nodes[i + 1];
            
            let connected = graph.edges.iter().any(|e| &e.from == from && &e.to == to);
            if !connected {
                complete = false;
                issues.push(format!("Gap between {} and {}", from, to));
            }
        }
        
        PathVerification {
            sound,
            complete,
            weakest_link,
            issues,
        }
    }
}

// ============================================================================
// PART 5: GRAPH OPERATIONS
// ============================================================================

impl FormalProofGraph {
    /// Create new empty proof graph
    pub fn new(problem_description: String) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            roots: HashSet::new(),
            goals: HashSet::new(),
            metadata: GraphMetadata {
                id: Uuid::new_v4().to_string(),
                created_at: Self::current_timestamp(),
                problem_description,
                total_nodes: 0,
                total_edges: 0,
                max_depth: 0,
            },
        }
    }
    
    /// Add node to graph
    pub fn add_node(&mut self, node: FormalProofNode) -> NodeId {
        let id = node.id.clone();
        
        // Update metadata
        if matches!(node.node_type, FormalNodeType::Axiom) {
            self.roots.insert(id.clone());
        }
        if matches!(node.node_type, FormalNodeType::Goal) {
            self.goals.insert(id.clone());
        }
        if node.depth > self.metadata.max_depth {
            self.metadata.max_depth = node.depth;
        }
        
        self.nodes.insert(id.clone(), node);
        self.metadata.total_nodes = self.nodes.len();
        
        id
    }
    
    /// Add edge to graph
    pub fn add_edge(&mut self, edge: FormalProofEdge) -> Result<String, String> {
        // Verify nodes exist
        if !self.nodes.contains_key(&edge.from) {
            return Err(format!("Source node {} not found", edge.from));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(format!("Target node {} not found", edge.to));
        }
        
        // Check for cycles
        if self.would_create_cycle(&edge.from, &edge.to) {
            return Err("Edge would create cycle".to_string());
        }
        
        let edge_id = edge.id.clone();
        
        // Update node degrees
        if let Some(from_node) = self.nodes.get_mut(&edge.from) {
            from_node.out_degree += 1;
        }
        if let Some(to_node) = self.nodes.get_mut(&edge.to) {
            to_node.in_degree += 1;
        }
        
        self.edges.push(edge);
        self.metadata.total_edges = self.edges.len();
        
        Ok(edge_id)
    }
    
    /// Check if adding edge would create cycle
    fn would_create_cycle(&self, from: &NodeId, to: &NodeId) -> bool {
        // BFS from 'to' to see if we can reach 'from'
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to.clone());
        
        while let Some(current) = queue.pop_front() {
            if &current == from {
                return true; // Found cycle
            }
            
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            
            // Add successors
            for edge in &self.edges {
                if &edge.from == &current {
                    queue.push_back(edge.to.clone());
                }
            }
        }
        
        false
    }
    
    /// Find all paths from root to goal
    pub fn find_paths(&self, from: &NodeId, to: &NodeId, max_length: usize) -> Vec<ProofPath> {
        let mut paths = Vec::new();
        let mut current_path = Vec::new();
        let mut current_edges = Vec::new();
        let mut visited = HashSet::new();
        
        self.dfs_paths(from, to, &mut current_path, &mut current_edges, &mut visited, &mut paths, max_length);
        
        paths
    }
    
    fn dfs_paths(
        &self,
        current: &NodeId,
        target: &NodeId,
        path: &mut Vec<NodeId>,
        edges: &mut Vec<String>,
        visited: &mut HashSet<NodeId>,
        results: &mut Vec<ProofPath>,
        max_length: usize,
    ) {
        if path.len() > max_length {
            return;
        }
        
        path.push(current.clone());
        visited.insert(current.clone());
        
        if current == target {
            // Found a path
            let mut proof_path = ProofPath {
                nodes: path.clone(),
                edges: edges.clone(),
                confidence: 0.0,
                length: path.len(),
                valid: false,
                verification: None,
            };
            proof_path.calculate_confidence(self);
            results.push(proof_path);
        } else {
            // Continue searching
            for edge in &self.edges {
                if &edge.from == current && !visited.contains(&edge.to) {
                    edges.push(edge.id.clone());
                    self.dfs_paths(&edge.to, target, path, edges, visited, results, max_length);
                    edges.pop();
                }
            }
        }
        
        path.pop();
        visited.remove(current);
    }
    
    /// Verify entire graph
    pub fn verify_graph(&mut self) -> GraphVerification {
        let mut sound = true;
        let mut complete = true;
        let mut issues = Vec::new();
        
        // Check all roots are axioms
        for root_id in &self.roots {
            if let Some(node) = self.nodes.get(root_id) {
                if !matches!(node.node_type, FormalNodeType::Axiom) {
                    sound = false;
                    issues.push(format!("Root {} is not an axiom", root_id));
                }
            }
        }
        
        // Check all goals are reachable from roots
        for goal_id in &self.goals {
            let mut reachable = false;
            for root_id in &self.roots {
                let paths = self.find_paths(root_id, goal_id, 20);
                if !paths.is_empty() {
                    reachable = true;
                    break;
                }
            }
            if !reachable {
                complete = false;
                issues.push(format!("Goal {} not reachable from any root", goal_id));
            }
        }
        
        // Check for orphaned nodes
        for (node_id, node) in &self.nodes {
            if node.in_degree == 0 && !self.roots.contains(node_id) {
                issues.push(format!("Orphaned node: {}", node_id));
            }
        }
        
        GraphVerification {
            sound,
            complete,
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            verified_nodes: self.nodes.values().filter(|n| n.verified).count(),
            verified_edges: self.edges.iter().filter(|e| e.verified).count(),
            issues,
        }
    }
    
    /// Calculate confidence for a conclusion
    pub fn calculate_conclusion_confidence(&self, goal_id: &NodeId) -> f64 {
        let mut max_confidence = 0.0;
        
        // Find all paths to goal
        for root_id in &self.roots {
            let paths = self.find_paths(root_id, goal_id, 20);
            for path in paths {
                if path.confidence > max_confidence {
                    max_confidence = path.confidence;
                }
            }
        }
        
        max_confidence
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphVerification {
    pub sound: bool,
    pub complete: bool,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub verified_nodes: usize,
    pub verified_edges: usize,
    pub issues: Vec<String>,
}

// ============================================================================
// PART 6: GRAPH ANALYSIS
// ============================================================================

impl FormalProofGraph {
    /// Find critical nodes (removal would disconnect graph)
    pub fn find_critical_nodes(&self) -> Vec<NodeId> {
        let mut critical = Vec::new();
        
        for (node_id, _) in &self.nodes {
            if self.is_critical_node(node_id) {
                critical.push(node_id.clone());
            }
        }
        
        critical
    }
    
    fn is_critical_node(&self, node_id: &NodeId) -> bool {
        // A node is critical if removing it disconnects any goal from roots
        // Simplified: check if it's on the only path to a goal
        
        for goal_id in &self.goals {
            let mut path_count = 0;
            for root_id in &self.roots {
                let paths = self.find_paths(root_id, goal_id, 20);
                for path in paths {
                    if path.nodes.contains(node_id) {
                        path_count += 1;
                    }
                }
            }
            
            // If this node is on all paths to a goal, it's critical
            if path_count > 0 {
                let total_paths: usize = self.roots.iter()
                    .map(|r| self.find_paths(r, goal_id, 20).len())
                    .sum();
                if path_count == total_paths {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Calculate graph complexity
    pub fn complexity(&self) -> f64 {
        let node_complexity = self.nodes.len() as f64;
        let edge_complexity = self.edges.len() as f64;
        let depth_complexity = self.metadata.max_depth as f64;
        
        // Weighted sum
        0.3 * node_complexity + 0.4 * edge_complexity + 0.3 * depth_complexity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_proof_graph_creation() {
        let mut graph = FormalProofGraph::new("Test problem".to_string());
        
        let axiom = FormalProofNode::new_axiom(
            NodeContent::Proposition("A is true".to_string()),
            1.0
        );
        let axiom_id = graph.add_node(axiom);
        
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.roots.contains(&axiom_id));
    }
    
    #[test]
    fn test_edge_addition() {
        let mut graph = FormalProofGraph::new("Test".to_string());
        
        let n1 = FormalProofNode::new_axiom(NodeContent::Proposition("A".to_string()), 1.0);
        let n2 = FormalProofNode::new_derived(NodeContent::Proposition("B".to_string()), 0.9, 1);
        
        let id1 = graph.add_node(n1);
        let id2 = graph.add_node(n2);
        
        let edge = FormalProofEdge::new(
            id1.clone(),
            id2.clone(),
            FormalInferenceRule::ModusPonens,
            0.95,
            Justification::Logical { steps: vec!["A implies B".to_string()] }
        );
        
        let result = graph.add_edge(edge);
        assert!(result.is_ok());
        assert_eq!(graph.edges.len(), 1);
    }
    
    #[test]
    fn test_cycle_detection() {
        let mut graph = FormalProofGraph::new("Test".to_string());
        
        let n1 = FormalProofNode::new_axiom(NodeContent::Proposition("A".to_string()), 1.0);
        let n2 = FormalProofNode::new_derived(NodeContent::Proposition("B".to_string()), 0.9, 1);
        
        let id1 = graph.add_node(n1);
        let id2 = graph.add_node(n2);
        
        // Add edge A -> B
        let edge1 = FormalProofEdge::new(
            id1.clone(),
            id2.clone(),
            FormalInferenceRule::ModusPonens,
            0.95,
            Justification::Logical { steps: vec![] }
        );
        graph.add_edge(edge1).unwrap();
        
        // Try to add edge B -> A (would create cycle)
        let edge2 = FormalProofEdge::new(
            id2.clone(),
            id1.clone(),
            FormalInferenceRule::ModusPonens,
            0.95,
            Justification::Logical { steps: vec![] }
        );
        
        let result = graph.add_edge(edge2);
        assert!(result.is_err());
    }
}
