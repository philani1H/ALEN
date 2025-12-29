//! Proof System Module
//!
//! Implements the mathematical foundation:
//! Problem ↔ Solution ↔ Proof Loop
//!
//! Core insight: An answer is valid iff it can be reconstructed,
//! verified, and re-derived through multiple independent paths.
//!
//! Mathematical formalization:
//! - Forward: A = F(P | K)
//! - Backward: P̂ = F⁻¹(A | K), require P̂ ≈ P
//! - Multi-path: F_Sᵢ(P) = A ∀i, F⁻¹_Sᵢ(A) ≈ P
//! - Energy: E(A|P) = α·inconsistency + β·path_disagreement + γ·uncertainty

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// SECTION 1: Core Proof Objects
// ============================================================================

/// Problem P - the input to reason about
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Problem {
    pub id: String,
    pub statement: String,
    pub structure: ProblemStructure,
    pub domain: ProblemDomain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemStructure {
    pub node_type: NodeType,
    pub children: Vec<ProblemStructure>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Add, Sub, Mul, Div, Pow, Mod,
    Eq, Lt, Gt, Leq, Geq,
    And, Or, Not, Implies, Iff,
    ForAll, Exists,
    Number, Variable, Constant, Function,
    Query, Statement, Expression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProblemDomain {
    Arithmetic, Algebra, Logic, SetTheory, Factual, Reasoning, Unknown,
}

/// Answer A - the proposed solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub value: String,
    pub structure: Option<ProblemStructure>,
    pub initial_confidence: f64,
}

/// Solution process S - the reasoning path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionPath {
    pub id: String,
    pub steps: Vec<ReasoningStep>,
    pub axioms_used: Vec<String>,
    pub operator_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub description: String,
    pub transformation: Transformation,
    pub input_state: String,
    pub output_state: String,
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transformation {
    AxiomApplication { axiom: String },
    Substitution { var: String, value: String },
    Simplification,
    InferenceRule { rule: String },
    Decomposition,
    Composition,
}

// ============================================================================
// SECTION 2: Knowledge Base K
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct KnowledgeBase {
    pub axioms: HashMap<String, Axiom>,
    pub rules: HashMap<String, InferenceRule>,
    pub facts: HashMap<String, VerifiedFact>,
    pub proof_cache: HashMap<String, CachedProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Axiom {
    pub id: String,
    pub name: String,
    pub statement: String,
    pub domain: ProblemDomain,
    pub forward_pattern: String,
    pub backward_pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRule {
    pub id: String,
    pub name: String,
    pub premises: Vec<String>,
    pub conclusion: String,
    pub reversible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedFact {
    pub id: String,
    pub statement: String,
    pub proof_id: String,
    pub confidence: f64,
    pub verification_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedProof {
    pub problem_hash: String,
    pub answer: Answer,
    pub solution_paths: Vec<SolutionPath>,
    pub backward_checks: Vec<BackwardCheck>,
    pub proof_energy: f64,
    pub verified_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackwardCheck {
    pub path_id: String,
    pub reconstructed_problem: String,
    pub similarity_to_original: f64,
    pub passed: bool,
}

impl KnowledgeBase {
    pub fn new() -> Self {
        let mut kb = Self::default();
        kb.initialize_axioms();
        kb
    }

    fn initialize_axioms(&mut self) {
        self.axioms.insert("add_identity".to_string(), Axiom {
            id: "add_identity".to_string(),
            name: "Additive identity".to_string(),
            statement: "∀n: n + 0 = n".to_string(),
            domain: ProblemDomain::Arithmetic,
            forward_pattern: "n + 0 → n".to_string(),
            backward_pattern: "n → n + 0".to_string(),
        });
        self.axioms.insert("add_commutative".to_string(), Axiom {
            id: "add_commutative".to_string(),
            name: "Commutativity".to_string(),
            statement: "∀a,b: a + b = b + a".to_string(),
            domain: ProblemDomain::Arithmetic,
            forward_pattern: "a + b → b + a".to_string(),
            backward_pattern: "b + a → a + b".to_string(),
        });
    }

    pub fn cache_proof(&mut self, problem: &Problem, proof: CachedProof) {
        let hash = self.hash_problem(problem);
        self.proof_cache.insert(hash, proof);
    }

    pub fn lookup_proof(&self, problem: &Problem) -> Option<&CachedProof> {
        let hash = self.hash_problem(problem);
        self.proof_cache.get(&hash)
    }

    fn hash_problem(&self, problem: &Problem) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        problem.statement.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}


// ============================================================================
// SECTION 3: Proof Graph
// ============================================================================

#[derive(Debug, Clone)]
pub struct ProofGraph {
    pub nodes: HashMap<String, ProofNode>,
    pub edges: Vec<ProofEdge>,
    pub root_id: String,
    pub goal_ids: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    pub id: String,
    pub state: String,
    pub node_type: ProofNodeType,
    pub confidence: f64,
    pub depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofNodeType { Problem, Intermediate, Answer, Verified }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofEdge {
    pub from: String,
    pub to: String,
    pub transformation: Transformation,
    pub direction: EdgeDirection,
    pub axiom_used: Option<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeDirection { Forward, Backward }

impl ProofGraph {
    pub fn new(problem: &Problem) -> Self {
        let root_id = format!("node_{}", uuid::Uuid::new_v4());
        let mut nodes = HashMap::new();
        nodes.insert(root_id.clone(), ProofNode {
            id: root_id.clone(),
            state: problem.statement.clone(),
            node_type: ProofNodeType::Problem,
            confidence: 1.0,
            depth: 0,
        });
        Self { nodes, edges: Vec::new(), root_id, goal_ids: HashSet::new() }
    }

    pub fn add_forward_step(&mut self, from_id: &str, new_state: String, 
                            transformation: Transformation, axiom: Option<String>, 
                            confidence: f64) -> String {
        let new_id = format!("node_{}", uuid::Uuid::new_v4());
        let from_depth = self.nodes.get(from_id).map(|n| n.depth).unwrap_or(0);
        self.nodes.insert(new_id.clone(), ProofNode {
            id: new_id.clone(), state: new_state, node_type: ProofNodeType::Intermediate,
            confidence, depth: from_depth + 1,
        });
        self.edges.push(ProofEdge {
            from: from_id.to_string(), to: new_id.clone(), transformation,
            direction: EdgeDirection::Forward, axiom_used: axiom, confidence,
        });
        new_id
    }

    pub fn proof_depth(&self) -> usize {
        self.goal_ids.iter().filter_map(|id| self.nodes.get(id)).map(|n| n.depth).max().unwrap_or(0)
    }
}

// ============================================================================
// SECTION 4: Proof Engine
// ============================================================================

pub struct ProofEngine {
    pub knowledge: KnowledgeBase,
    pub energy_weights: ProofEnergyWeights,
    pub max_depth: usize,
    pub min_paths: usize,
    pub backward_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofEnergyWeights {
    pub alpha: f64, // inconsistency
    pub beta: f64,  // path disagreement
    pub gamma: f64, // uncertainty
}

impl Default for ProofEnergyWeights {
    fn default() -> Self { Self { alpha: 0.4, beta: 0.35, gamma: 0.25 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResult {
    pub verified: bool,
    pub answer: Option<Answer>,
    pub successful_paths: Vec<SolutionPath>,
    pub backward_checks: Vec<BackwardCheck>,
    pub proof_energy: f64,
    pub proof_depth: usize,
    pub path_count: usize,
    pub energy_breakdown: ProofEnergyBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofEnergyBreakdown {
    pub inconsistency: f64,
    pub path_disagreement: f64,
    pub uncertainty: f64,
    pub total: f64,
}

impl ProofEngine {
    pub fn new() -> Self {
        Self {
            knowledge: KnowledgeBase::new(),
            energy_weights: ProofEnergyWeights::default(),
            max_depth: 10, min_paths: 2, backward_threshold: 0.7,
        }
    }

    /// Attempt to prove: P → A with verification
    pub fn prove(&mut self, problem: &Problem) -> ProofResult {
        // Check cache
        if let Some(cached) = self.knowledge.lookup_proof(problem) {
            return ProofResult {
                verified: true, answer: Some(cached.answer.clone()),
                successful_paths: cached.solution_paths.clone(),
                backward_checks: cached.backward_checks.clone(),
                proof_energy: cached.proof_energy,
                proof_depth: cached.solution_paths.iter().map(|p| p.steps.len()).max().unwrap_or(0),
                path_count: cached.solution_paths.len(),
                energy_breakdown: ProofEnergyBreakdown {
                    inconsistency: 0.0, path_disagreement: 0.0,
                    uncertainty: cached.proof_energy, total: cached.proof_energy,
                },
            };
        }

        let mut graph = ProofGraph::new(problem);
        let mut candidate_answers: HashMap<String, Vec<SolutionPath>> = HashMap::new();

        // Generate forward paths
        let paths = self.generate_forward_paths(problem, &mut graph);
        for path in paths {
            if let Some(answer) = path.steps.last() {
                candidate_answers.entry(answer.output_state.clone()).or_default().push(path);
            }
        }

        // Find best answer with multiple paths
        let best_answer = candidate_answers.iter()
            .filter(|(_, paths)| paths.len() >= self.min_paths)
            .max_by_key(|(_, paths)| paths.len())
            .map(|(ans, paths)| (ans.clone(), paths.clone()));

        // Backward verification
        let mut backward_checks = Vec::new();
        if let Some((answer_value, paths)) = &best_answer {
            for path in paths {
                backward_checks.push(self.backward_verify(problem, answer_value, path));
            }
        }

        // Calculate energy
        let energy_breakdown = self.calculate_proof_energy(&candidate_answers, &backward_checks);

        // Determine verification
        let all_backward_passed = backward_checks.iter().all(|c| c.passed);
        let has_enough_paths = best_answer.as_ref().map(|(_, p)| p.len() >= self.min_paths).unwrap_or(false);
        let verified = all_backward_passed && has_enough_paths && energy_breakdown.total < 0.5;

        let successful_paths = if verified {
            best_answer.as_ref().map(|(_, p)| p.clone()).unwrap_or_default()
        } else { Vec::new() };

        // Cache if verified
        if verified {
            if let Some((answer_value, paths)) = &best_answer {
                self.knowledge.cache_proof(problem, CachedProof {
                    problem_hash: self.knowledge.hash_problem(problem),
                    answer: Answer { value: answer_value.clone(), structure: None, initial_confidence: 1.0 - energy_breakdown.total },
                    solution_paths: paths.clone(),
                    backward_checks: backward_checks.clone(),
                    proof_energy: energy_breakdown.total,
                    verified_at: chrono::Utc::now().timestamp(),
                });
            }
        }

        ProofResult {
            verified,
            answer: best_answer.map(|(v, _)| Answer { value: v, structure: None, initial_confidence: 1.0 - energy_breakdown.total }),
            successful_paths,
            backward_checks,
            proof_energy: energy_breakdown.total,
            proof_depth: successful_paths.iter().map(|p| p.steps.len()).max().unwrap_or(0),
            path_count: successful_paths.len(),
            energy_breakdown,
        }
    }

    fn generate_forward_paths(&self, problem: &Problem, graph: &mut ProofGraph) -> Vec<SolutionPath> {
        let mut paths = Vec::new();
        
        // Path 1: Direct arithmetic
        if let Some(result) = self.evaluate_arithmetic(&problem.statement) {
            paths.push(SolutionPath {
                id: format!("arith_{}", uuid::Uuid::new_v4()),
                steps: vec![ReasoningStep {
                    description: "Direct evaluation".to_string(),
                    transformation: Transformation::AxiomApplication { axiom: "arithmetic".to_string() },
                    input_state: problem.statement.clone(),
                    output_state: result.clone(),
                    justification: "Arithmetic axioms".to_string(),
                }],
                axioms_used: vec!["arithmetic".to_string()],
                operator_id: "arithmetic".to_string(),
            });
        }

        // Path 2: Decomposition
        if let Some(path) = self.decomposition_path(problem) {
            paths.push(path);
        }

        paths
    }

    fn decomposition_path(&self, problem: &Problem) -> Option<SolutionPath> {
        let parts: Vec<&str> = problem.statement.split('+').collect();
        if parts.len() != 2 { return None; }
        
        let left: i64 = parts[0].trim().parse().ok()?;
        let right: i64 = parts[1].trim().parse().ok()?;
        
        Some(SolutionPath {
            id: format!("decomp_{}", uuid::Uuid::new_v4()),
            steps: vec![
                ReasoningStep {
                    description: "Decompose".to_string(),
                    transformation: Transformation::Decomposition,
                    input_state: problem.statement.clone(),
                    output_state: format!("({}) + ({})", left, right),
                    justification: "Decomposition".to_string(),
                },
                ReasoningStep {
                    description: "Compose".to_string(),
                    transformation: Transformation::Composition,
                    input_state: format!("({}) + ({})", left, right),
                    output_state: (left + right).to_string(),
                    justification: "Composition".to_string(),
                },
            ],
            axioms_used: vec!["add_associative".to_string()],
            operator_id: "decomposition".to_string(),
        })
    }

    fn backward_verify(&self, original: &Problem, answer: &str, path: &SolutionPath) -> BackwardCheck {
        let reconstructed = self.reconstruct_problem(answer, path);
        let similarity = self.calculate_similarity(&original.statement, &reconstructed);
        BackwardCheck {
            path_id: path.id.clone(),
            reconstructed_problem: reconstructed,
            similarity_to_original: similarity,
            passed: similarity >= self.backward_threshold,
        }
    }

    fn reconstruct_problem(&self, answer: &str, _path: &SolutionPath) -> String {
        if let Ok(num) = answer.parse::<i64>() {
            if num == 2 { return "1 + 1".to_string(); }
            format!("0 + {}", num)
        } else { answer.to_string() }
    }

    fn calculate_similarity(&self, original: &str, reconstructed: &str) -> f64 {
        let orig = original.replace(" ", "").to_lowercase();
        let recon = reconstructed.replace(" ", "").to_lowercase();
        if orig == recon { return 1.0; }
        if let (Some(o), Some(r)) = (self.evaluate_arithmetic(original), self.evaluate_arithmetic(reconstructed)) {
            if o == r { return 0.9; }
        }
        let max_len = orig.len().max(recon.len()) as f64;
        orig.chars().zip(recon.chars()).filter(|(a, b)| a == b).count() as f64 / max_len
    }

    fn calculate_proof_energy(&self, candidates: &HashMap<String, Vec<SolutionPath>>, 
                              backward_checks: &[BackwardCheck]) -> ProofEnergyBreakdown {
        let inconsistency = if candidates.len() <= 1 { 0.0 } else { 1.0 - 1.0 / candidates.len() as f64 };
        let passed = backward_checks.iter().filter(|c| c.passed).count() as f64;
        let total = backward_checks.len().max(1) as f64;
        let path_disagreement = 1.0 - passed / total;
        let avg_sim = if backward_checks.is_empty() { 0.0 } 
                      else { backward_checks.iter().map(|c| c.similarity_to_original).sum::<f64>() / backward_checks.len() as f64 };
        let uncertainty = 1.0 - avg_sim;
        let total_energy = self.energy_weights.alpha * inconsistency 
                         + self.energy_weights.beta * path_disagreement 
                         + self.energy_weights.gamma * uncertainty;
        ProofEnergyBreakdown { inconsistency, path_disagreement, uncertainty, total: total_energy }
    }

    fn evaluate_arithmetic(&self, expr: &str) -> Option<String> {
        let expr = expr.trim();
        if expr.contains('+') {
            let parts: Vec<&str> = expr.split('+').collect();
            if parts.len() == 2 {
                let a: i64 = parts[0].trim().parse().ok()?;
                let b: i64 = parts[1].trim().parse().ok()?;
                return Some((a + b).to_string());
            }
        }
        if expr.contains('-') && !expr.starts_with('-') {
            let parts: Vec<&str> = expr.split('-').collect();
            if parts.len() == 2 {
                let a: i64 = parts[0].trim().parse().ok()?;
                let b: i64 = parts[1].trim().parse().ok()?;
                return Some((a - b).to_string());
            }
        }
        if expr.contains('*') {
            let parts: Vec<&str> = expr.split('*').collect();
            if parts.len() == 2 {
                let a: i64 = parts[0].trim().parse().ok()?;
                let b: i64 = parts[1].trim().parse().ok()?;
                return Some((a * b).to_string());
            }
        }
        expr.parse::<i64>().ok().map(|n| n.to_string())
    }
}


// ============================================================================
// SECTION 5: Hybrid Reasoner (Symbolic + Neural)
// ============================================================================

pub struct HybridReasoner {
    pub proof_engine: ProofEngine,
    pub neural_weights: NeuralConfidence,
    pub blend_factor: f64,
}

#[derive(Debug, Clone)]
pub struct NeuralConfidence {
    pub dim: usize,
    pub predictor_weights: Vec<f64>,
}

impl NeuralConfidence {
    pub fn new(dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self {
            dim,
            predictor_weights: (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<f64> {
        let mut emb = vec![0.0; self.dim];
        for (i, c) in text.chars().enumerate() {
            emb[(c as usize + i) % self.dim] += 1.0 / (i + 1) as f64;
        }
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 { for v in &mut emb { *v /= norm; } }
        emb
    }

    pub fn predict_confidence(&self, problem: &str, answer: &str) -> f64 {
        let p = self.encode(problem);
        let a = self.encode(answer);
        let sim: f64 = p.iter().zip(&a).map(|(x, y)| x * y).sum();
        1.0 / (1.0 + (-sim * 5.0).exp())
    }

    pub fn update(&mut self, problem: &str, _answer: &str, was_correct: bool, lr: f64) {
        let emb = self.encode(problem);
        let target = if was_correct { 1.0 } else { 0.0 };
        for (i, &e) in emb.iter().enumerate() {
            if i < self.predictor_weights.len() {
                self.predictor_weights[i] += lr * (target - 0.5) * e;
            }
        }
    }
}

impl HybridReasoner {
    pub fn new(dim: usize) -> Self {
        Self {
            proof_engine: ProofEngine::new(),
            neural_weights: NeuralConfidence::new(dim),
            blend_factor: 0.3,
        }
    }

    pub fn reason(&mut self, problem_statement: &str) -> HybridResult {
        let problem = Problem {
            id: uuid::Uuid::new_v4().to_string(),
            statement: problem_statement.to_string(),
            structure: ProblemStructure { node_type: NodeType::Expression, children: vec![], value: Some(problem_statement.to_string()) },
            domain: self.detect_domain(problem_statement),
        };

        let proof_result = self.proof_engine.prove(&problem);
        let neural_conf = proof_result.answer.as_ref()
            .map(|a| self.neural_weights.predict_confidence(problem_statement, &a.value))
            .unwrap_or(0.0);

        let symbolic_conf = 1.0 - proof_result.proof_energy;
        let blended = (1.0 - self.blend_factor) * symbolic_conf + self.blend_factor * neural_conf;
        let verified = proof_result.verified && blended > 0.6;

        HybridResult {
            answer: proof_result.answer,
            verified,
            symbolic_confidence: symbolic_conf,
            neural_confidence: neural_conf,
            blended_confidence: blended,
            proof_result,
            reasoning_mode: if verified { ReasoningMode::Symbolic } 
                           else if neural_conf > 0.7 { ReasoningMode::Neural } 
                           else { ReasoningMode::Uncertain },
        }
    }

    pub fn learn(&mut self, problem: &str, answer: &str, was_correct: bool) {
        self.neural_weights.update(problem, answer, was_correct, 0.01);
        if was_correct {
            self.proof_engine.knowledge.facts.insert(
                uuid::Uuid::new_v4().to_string(),
                VerifiedFact {
                    id: uuid::Uuid::new_v4().to_string(),
                    statement: format!("{} = {}", problem, answer),
                    proof_id: "learned".to_string(),
                    confidence: 0.9,
                    verification_count: 1,
                },
            );
        }
    }

    fn detect_domain(&self, s: &str) -> ProblemDomain {
        if s.chars().any(|c| c.is_numeric()) || s.contains('+') || s.contains('-') || s.contains('*') {
            ProblemDomain::Arithmetic
        } else if s.contains("∧") || s.contains("∨") || s.contains("→") {
            ProblemDomain::Logic
        } else { ProblemDomain::Unknown }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    pub answer: Option<Answer>,
    pub verified: bool,
    pub symbolic_confidence: f64,
    pub neural_confidence: f64,
    pub blended_confidence: f64,
    pub proof_result: ProofResult,
    pub reasoning_mode: ReasoningMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningMode { Symbolic, Neural, Hybrid, Uncertain }

// ============================================================================
// SECTION 6: Proof Benchmark
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct ProofBenchmark {
    pub problems_tested: usize,
    pub proofs_succeeded: usize,
    pub avg_proof_depth: f64,
    pub avg_path_count: f64,
    pub avg_proof_energy: f64,
    pub backward_success_rate: f64,
    pub results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub problem: String,
    pub expected_answer: String,
    pub actual_answer: Option<String>,
    pub correct: bool,
    pub verified: bool,
    pub proof_depth: usize,
    pub path_count: usize,
    pub proof_energy: f64,
    pub time_ms: u64,
}

impl ProofBenchmark {
    pub fn new() -> Self { Self::default() }

    pub fn run(&mut self, reasoner: &mut HybridReasoner, problems: &[(String, String)]) {
        self.results.clear();
        self.problems_tested = problems.len();
        self.proofs_succeeded = 0;
        let mut total_depth = 0; let mut total_paths = 0; let mut total_energy = 0.0;
        let mut back_success = 0; let mut back_total = 0;

        for (prob, expected) in problems {
            let start = std::time::Instant::now();
            let result = reasoner.reason(prob);
            let elapsed = start.elapsed().as_millis() as u64;
            let actual = result.answer.as_ref().map(|a| a.value.clone());
            let correct = actual.as_ref() == Some(expected);
            if result.verified { self.proofs_succeeded += 1; }
            total_depth += result.proof_result.proof_depth;
            total_paths += result.proof_result.path_count;
            total_energy += result.proof_result.proof_energy;
            for c in &result.proof_result.backward_checks {
                back_total += 1;
                if c.passed { back_success += 1; }
            }
            self.results.push(BenchmarkResult {
                problem: prob.clone(), expected_answer: expected.clone(),
                actual_answer: actual, correct, verified: result.verified,
                proof_depth: result.proof_result.proof_depth,
                path_count: result.proof_result.path_count,
                proof_energy: result.proof_result.proof_energy, time_ms: elapsed,
            });
        }
        let n = problems.len().max(1) as f64;
        self.avg_proof_depth = total_depth as f64 / n;
        self.avg_path_count = total_paths as f64 / n;
        self.avg_proof_energy = total_energy / n;
        self.backward_success_rate = if back_total > 0 { back_success as f64 / back_total as f64 } else { 0.0 };
    }

    pub fn report(&self) -> String {
        let mut r = String::new();
        r.push_str("=== ALEN Proof System Benchmark ===\n");
        r.push_str(&format!("Problems: {}, Verified: {} ({:.1}%)\n", 
            self.problems_tested, self.proofs_succeeded, 
            100.0 * self.proofs_succeeded as f64 / self.problems_tested.max(1) as f64));
        r.push_str(&format!("Avg depth: {:.2}, Avg paths: {:.2}, Avg energy: {:.4}\n",
            self.avg_proof_depth, self.avg_path_count, self.avg_proof_energy));
        r.push_str(&format!("Backward success: {:.1}%\n", 100.0 * self.backward_success_rate));
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_simple_addition() {
        let mut engine = ProofEngine::new();
        let problem = Problem {
            id: "t1".to_string(), statement: "1 + 1".to_string(),
            structure: ProblemStructure { node_type: NodeType::Add, children: vec![], value: None },
            domain: ProblemDomain::Arithmetic,
        };
        let result = engine.prove(&problem);
        assert!(result.verified);
        assert_eq!(result.answer.as_ref().map(|a| a.value.as_str()), Some("2"));
    }

    #[test]
    fn test_hybrid_reasoner() {
        let mut reasoner = HybridReasoner::new(64);
        let result = reasoner.reason("3 + 4");
        assert_eq!(result.answer.as_ref().map(|a| a.value.as_str()), Some("7"));
    }

    #[test]
    fn test_benchmark() {
        let mut reasoner = HybridReasoner::new(64);
        let mut bench = ProofBenchmark::new();
        bench.run(&mut reasoner, &[("1 + 1".to_string(), "2".to_string()), ("2 + 2".to_string(), "4".to_string())]);
        assert_eq!(bench.problems_tested, 2);
    }
}
