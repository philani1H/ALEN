//! ALEN Comprehensive Knowledge Base
//!
//! Contains training data for:
//! - Physics (mechanics, thermodynamics, electromagnetism, quantum, relativity)
//! - Mathematics (algebra, calculus, linear algebra, statistics, logic)
//! - Language (grammar, semantics, syntax, vocabulary)
//! - Computer Science (algorithms, data structures, complexity)
//! - Chemistry and Biology basics
//! - Logic and Reasoning patterns

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A training example with input, expected output, and verification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeItem {
    /// Category of knowledge
    pub category: KnowledgeCategory,
    /// Subcategory for finer classification
    pub subcategory: String,
    /// The question or problem
    pub input: String,
    /// The correct answer
    pub output: String,
    /// Explanation of reasoning (for verification)
    pub reasoning: String,
    /// Backward verification: what question does this answer imply?
    pub backward_check: String,
    /// Related concepts
    pub related: Vec<String>,
    /// Difficulty level (1-10)
    pub difficulty: u8,
    /// Prerequisites
    pub prerequisites: Vec<String>,
}

/// Knowledge categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum KnowledgeCategory {
    Physics,
    Mathematics,
    Language,
    ComputerScience,
    Chemistry,
    Biology,
    Logic,
    Philosophy,
    History,
    Geography,
    Economics,
    Psychology,
}

impl std::fmt::Display for KnowledgeCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KnowledgeCategory::Physics => write!(f, "physics"),
            KnowledgeCategory::Mathematics => write!(f, "mathematics"),
            KnowledgeCategory::Language => write!(f, "language"),
            KnowledgeCategory::ComputerScience => write!(f, "computer_science"),
            KnowledgeCategory::Chemistry => write!(f, "chemistry"),
            KnowledgeCategory::Biology => write!(f, "biology"),
            KnowledgeCategory::Logic => write!(f, "logic"),
            KnowledgeCategory::Philosophy => write!(f, "philosophy"),
            KnowledgeCategory::History => write!(f, "history"),
            KnowledgeCategory::Geography => write!(f, "geography"),
            KnowledgeCategory::Economics => write!(f, "economics"),
            KnowledgeCategory::Psychology => write!(f, "psychology"),
        }
    }
}

/// The comprehensive knowledge base
pub struct KnowledgeBase {
    items: Vec<KnowledgeItem>,
    by_category: HashMap<KnowledgeCategory, Vec<usize>>,
}

impl KnowledgeBase {
    /// Create a new knowledge base with all built-in knowledge
    pub fn new() -> Self {
        let mut kb = Self {
            items: Vec::new(),
            by_category: HashMap::new(),
        };
        
        // Load all knowledge
        kb.load_physics();
        kb.load_mathematics();
        kb.load_language();
        kb.load_computer_science();
        kb.load_chemistry();
        kb.load_biology();
        kb.load_logic();
        
        kb
    }

    fn add_item(&mut self, item: KnowledgeItem) {
        let idx = self.items.len();
        let cat = item.category;
        self.items.push(item);
        self.by_category.entry(cat).or_insert_with(Vec::new).push(idx);
    }

    /// Get all items
    pub fn all_items(&self) -> &[KnowledgeItem] {
        &self.items
    }

    /// Get items by category
    pub fn by_category(&self, cat: KnowledgeCategory) -> Vec<&KnowledgeItem> {
        self.by_category.get(&cat)
            .map(|indices| indices.iter().map(|&i| &self.items[i]).collect())
            .unwrap_or_default()
    }

    /// Get item count
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    // =========================================================================
    // PHYSICS KNOWLEDGE
    // =========================================================================
    fn load_physics(&mut self) {
        // Classical Mechanics
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "classical_mechanics".into(),
            input: "What is Newton's First Law of Motion?".into(),
            output: "An object at rest stays at rest, and an object in motion stays in motion with the same speed and direction, unless acted upon by an unbalanced force.".into(),
            reasoning: "This law describes inertia - the tendency of objects to resist changes in their state of motion. Without external forces, velocity remains constant.".into(),
            backward_check: "What law describes the concept of inertia and why objects maintain constant velocity without external forces?".into(),
            related: vec!["inertia".into(), "force".into(), "velocity".into(), "acceleration".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "classical_mechanics".into(),
            input: "What is the formula for Newton's Second Law?".into(),
            output: "F = ma, where F is force, m is mass, and a is acceleration.".into(),
            reasoning: "Force equals mass times acceleration. This shows that acceleration is proportional to force and inversely proportional to mass.".into(),
            backward_check: "What equation relates force, mass, and acceleration?".into(),
            related: vec!["force".into(), "mass".into(), "acceleration".into(), "momentum".into()],
            difficulty: 2,
            prerequisites: vec!["Newton's First Law".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "classical_mechanics".into(),
            input: "What is the formula for kinetic energy?".into(),
            output: "KE = ½mv², where m is mass and v is velocity.".into(),
            reasoning: "Kinetic energy is the energy of motion. It increases with the square of velocity, meaning doubling speed quadruples kinetic energy.".into(),
            backward_check: "What formula calculates the energy an object possesses due to its motion?".into(),
            related: vec!["energy".into(), "work".into(), "momentum".into(), "velocity".into()],
            difficulty: 3,
            prerequisites: vec!["velocity".into(), "mass".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "classical_mechanics".into(),
            input: "What is the formula for gravitational potential energy?".into(),
            output: "PE = mgh, where m is mass, g is gravitational acceleration (9.8 m/s²), and h is height.".into(),
            reasoning: "Potential energy is stored energy due to position. Higher objects have more potential to do work when falling.".into(),
            backward_check: "How do you calculate the stored energy of an object based on its height?".into(),
            related: vec!["gravity".into(), "energy".into(), "work".into(), "height".into()],
            difficulty: 3,
            prerequisites: vec!["gravity".into(), "energy".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "classical_mechanics".into(),
            input: "What is the law of conservation of energy?".into(),
            output: "Energy cannot be created or destroyed, only transformed from one form to another. The total energy in an isolated system remains constant.".into(),
            reasoning: "This fundamental principle means that while energy changes form (kinetic to potential, etc.), the total amount is always conserved.".into(),
            backward_check: "What law states that the total energy in an isolated system remains constant?".into(),
            related: vec!["energy".into(), "thermodynamics".into(), "work".into()],
            difficulty: 4,
            prerequisites: vec!["kinetic energy".into(), "potential energy".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "classical_mechanics".into(),
            input: "What is momentum and its formula?".into(),
            output: "Momentum is the product of mass and velocity: p = mv. It represents the quantity of motion an object has.".into(),
            reasoning: "Momentum is conserved in collisions. A heavier or faster object has more momentum and is harder to stop.".into(),
            backward_check: "What physical quantity equals mass times velocity and is conserved in collisions?".into(),
            related: vec!["mass".into(), "velocity".into(), "collision".into(), "impulse".into()],
            difficulty: 3,
            prerequisites: vec!["mass".into(), "velocity".into()],
        });

        // Thermodynamics
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "thermodynamics".into(),
            input: "What is the First Law of Thermodynamics?".into(),
            output: "The change in internal energy of a system equals heat added minus work done by the system: ΔU = Q - W.".into(),
            reasoning: "This is conservation of energy applied to thermal systems. Energy input as heat either increases internal energy or does work.".into(),
            backward_check: "What thermodynamic law relates internal energy change to heat and work?".into(),
            related: vec!["energy".into(), "heat".into(), "work".into(), "entropy".into()],
            difficulty: 5,
            prerequisites: vec!["energy conservation".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "thermodynamics".into(),
            input: "What is the Second Law of Thermodynamics?".into(),
            output: "In any natural process, the total entropy of an isolated system always increases or remains constant; it never decreases. Heat flows spontaneously from hot to cold.".into(),
            reasoning: "This law explains why processes have a preferred direction. Disorder (entropy) naturally increases over time.".into(),
            backward_check: "What law explains why heat flows from hot to cold and entropy increases?".into(),
            related: vec!["entropy".into(), "heat".into(), "reversibility".into(), "Carnot".into()],
            difficulty: 6,
            prerequisites: vec!["First Law of Thermodynamics".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "thermodynamics".into(),
            input: "What is entropy?".into(),
            output: "Entropy is a measure of disorder or randomness in a system. It quantifies the number of microscopic configurations that correspond to a thermodynamic system's macroscopic state.".into(),
            reasoning: "Higher entropy means more possible arrangements. Systems tend toward maximum entropy (most probable state).".into(),
            backward_check: "What thermodynamic quantity measures disorder and always increases in isolated systems?".into(),
            related: vec!["thermodynamics".into(), "disorder".into(), "probability".into(), "information".into()],
            difficulty: 7,
            prerequisites: vec!["Second Law of Thermodynamics".into()],
        });

        // Electromagnetism
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "electromagnetism".into(),
            input: "What is Coulomb's Law?".into(),
            output: "F = kq₁q₂/r², where k ≈ 8.99×10⁹ N·m²/C². The force between two charges is proportional to their product and inversely proportional to the square of distance.".into(),
            reasoning: "Like charges repel, unlike charges attract. The force decreases rapidly with distance (inverse square law).".into(),
            backward_check: "What law describes the force between two electric charges?".into(),
            related: vec!["charge".into(), "electric field".into(), "force".into()],
            difficulty: 4,
            prerequisites: vec!["force".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "electromagnetism".into(),
            input: "What is Ohm's Law?".into(),
            output: "V = IR, where V is voltage, I is current, and R is resistance.".into(),
            reasoning: "Voltage drives current through resistance. Higher resistance means less current for the same voltage.".into(),
            backward_check: "What equation relates voltage, current, and resistance in a circuit?".into(),
            related: vec!["voltage".into(), "current".into(), "resistance".into(), "power".into()],
            difficulty: 3,
            prerequisites: vec!["electric charge".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "electromagnetism".into(),
            input: "What is electromagnetic induction?".into(),
            output: "A changing magnetic field induces an electric field (and thus an EMF/voltage) in a conductor. Described by Faraday's Law: EMF = -dΦ/dt.".into(),
            reasoning: "This is how generators work - rotating magnets induce current. The negative sign (Lenz's Law) shows induced current opposes the change.".into(),
            backward_check: "What phenomenon describes how changing magnetic fields create electric fields?".into(),
            related: vec!["Faraday".into(), "magnetic field".into(), "electric field".into(), "generator".into()],
            difficulty: 6,
            prerequisites: vec!["magnetic field".into(), "electric field".into()],
        });

        // Quantum Mechanics
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "quantum_mechanics".into(),
            input: "What is the Heisenberg Uncertainty Principle?".into(),
            output: "It is impossible to simultaneously know both the exact position and exact momentum of a particle: ΔxΔp ≥ ℏ/2.".into(),
            reasoning: "This is a fundamental limit of nature, not measurement error. Particles don't have definite position AND momentum simultaneously.".into(),
            backward_check: "What principle states that position and momentum cannot be precisely known simultaneously?".into(),
            related: vec!["quantum".into(), "wave function".into(), "measurement".into(), "Planck".into()],
            difficulty: 8,
            prerequisites: vec!["wave-particle duality".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "quantum_mechanics".into(),
            input: "What is wave-particle duality?".into(),
            output: "All matter and energy exhibits both wave-like and particle-like properties. Light behaves as waves (interference) and particles (photons). Electrons show wave patterns in double-slit experiments.".into(),
            reasoning: "This duality is fundamental to quantum mechanics. What we observe depends on the type of measurement we make.".into(),
            backward_check: "What concept describes how quantum objects behave as both waves and particles?".into(),
            related: vec!["photon".into(), "electron".into(), "interference".into(), "de Broglie".into()],
            difficulty: 7,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "quantum_mechanics".into(),
            input: "What is quantum superposition?".into(),
            output: "A quantum system can exist in multiple states simultaneously until measured. The wave function describes probabilities of all possible states.".into(),
            reasoning: "Before measurement, a particle doesn't have a definite state - it's in a superposition. Measurement 'collapses' this to one outcome.".into(),
            backward_check: "What quantum phenomenon allows particles to exist in multiple states simultaneously?".into(),
            related: vec!["wave function".into(), "measurement".into(), "Schrödinger".into(), "collapse".into()],
            difficulty: 8,
            prerequisites: vec!["wave-particle duality".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "quantum_mechanics".into(),
            input: "What is Schrödinger's equation?".into(),
            output: "iℏ∂ψ/∂t = Ĥψ. It describes how the quantum state (wave function ψ) of a system evolves over time, where Ĥ is the Hamiltonian operator.".into(),
            reasoning: "This is the fundamental equation of quantum mechanics. It's deterministic for wave function evolution, but ψ gives probabilities.".into(),
            backward_check: "What equation governs the time evolution of quantum wave functions?".into(),
            related: vec!["wave function".into(), "Hamiltonian".into(), "quantum state".into()],
            difficulty: 9,
            prerequisites: vec!["superposition".into(), "operators".into()],
        });

        // Relativity
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "relativity".into(),
            input: "What is Einstein's special theory of relativity?".into(),
            output: "The laws of physics are the same in all inertial reference frames. The speed of light (c ≈ 3×10⁸ m/s) is constant for all observers regardless of their motion.".into(),
            reasoning: "This leads to time dilation, length contraction, and mass-energy equivalence. Space and time are relative, spacetime is absolute.".into(),
            backward_check: "What theory states that the speed of light is constant for all observers?".into(),
            related: vec!["time dilation".into(), "length contraction".into(), "E=mc²".into()],
            difficulty: 7,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "relativity".into(),
            input: "What is E=mc²?".into(),
            output: "Energy equals mass times the speed of light squared. This shows mass and energy are equivalent - a small amount of mass contains enormous energy.".into(),
            reasoning: "This explains nuclear energy. The sun converts 4 million tons of mass to energy every second via fusion.".into(),
            backward_check: "What equation shows the equivalence of mass and energy?".into(),
            related: vec!["mass".into(), "energy".into(), "nuclear".into(), "Einstein".into()],
            difficulty: 5,
            prerequisites: vec!["special relativity".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "relativity".into(),
            input: "What is time dilation?".into(),
            output: "Time passes slower for objects moving at high speeds relative to a stationary observer, or in stronger gravitational fields. t' = t/√(1-v²/c²).".into(),
            reasoning: "GPS satellites must account for time dilation (both special and general relativity effects) to maintain accuracy.".into(),
            backward_check: "What relativistic effect causes moving clocks to run slower?".into(),
            related: vec!["special relativity".into(), "general relativity".into(), "spacetime".into()],
            difficulty: 7,
            prerequisites: vec!["special relativity".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "relativity".into(),
            input: "What is general relativity?".into(),
            output: "Gravity is not a force but the curvature of spacetime caused by mass and energy. Objects follow geodesics (straightest paths) in curved spacetime.".into(),
            reasoning: "Massive objects bend spacetime. Light bends around the sun, black holes warp spacetime extremely, and GPS requires relativistic corrections.".into(),
            backward_check: "What theory describes gravity as the curvature of spacetime?".into(),
            related: vec!["gravity".into(), "spacetime".into(), "black hole".into(), "geodesic".into()],
            difficulty: 9,
            prerequisites: vec!["special relativity".into()],
        });

        // Waves and Optics
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "waves".into(),
            input: "What is the wave equation?".into(),
            output: "v = fλ, where v is wave speed, f is frequency, and λ is wavelength.".into(),
            reasoning: "All waves satisfy this relationship. Higher frequency means shorter wavelength for a given medium.".into(),
            backward_check: "What equation relates wave speed, frequency, and wavelength?".into(),
            related: vec!["frequency".into(), "wavelength".into(), "sound".into(), "light".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Physics,
            subcategory: "waves".into(),
            input: "What is the Doppler effect?".into(),
            output: "The apparent change in frequency of a wave when the source and observer are moving relative to each other. Approaching sources have higher frequency (blue shift), receding sources have lower frequency (red shift).".into(),
            reasoning: "Used in radar, medical ultrasound, and measuring cosmic expansion. The universe is expanding because distant galaxies are red-shifted.".into(),
            backward_check: "What effect causes the pitch of a siren to change as it passes you?".into(),
            related: vec!["waves".into(), "frequency".into(), "redshift".into(), "cosmology".into()],
            difficulty: 5,
            prerequisites: vec!["wave equation".into()],
        });
    }

    // =========================================================================
    // MATHEMATICS KNOWLEDGE
    // =========================================================================
    fn load_mathematics(&mut self) {
        // Arithmetic and Algebra
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "arithmetic".into(),
            input: "What is 2 + 2?".into(),
            output: "4".into(),
            reasoning: "Basic addition: combining two units with two more units gives four units total.".into(),
            backward_check: "What two equal numbers sum to 4?".into(),
            related: vec!["addition".into(), "numbers".into()],
            difficulty: 1,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "arithmetic".into(),
            input: "What is multiplication?".into(),
            output: "Multiplication is repeated addition. a × b means adding 'a' to itself 'b' times.".into(),
            reasoning: "3 × 4 = 3 + 3 + 3 + 3 = 12. It's also the area of a rectangle with sides a and b.".into(),
            backward_check: "What operation represents repeated addition?".into(),
            related: vec!["addition".into(), "division".into(), "factors".into()],
            difficulty: 1,
            prerequisites: vec!["addition".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "algebra".into(),
            input: "What is the quadratic formula?".into(),
            output: "x = (-b ± √(b²-4ac)) / 2a for the equation ax² + bx + c = 0.".into(),
            reasoning: "Derived by completing the square. The discriminant (b²-4ac) determines if roots are real, equal, or complex.".into(),
            backward_check: "What formula gives the solutions to ax² + bx + c = 0?".into(),
            related: vec!["polynomial".into(), "roots".into(), "discriminant".into()],
            difficulty: 4,
            prerequisites: vec!["algebra basics".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "algebra".into(),
            input: "What are the laws of exponents?".into(),
            output: "a^m × a^n = a^(m+n), a^m / a^n = a^(m-n), (a^m)^n = a^(mn), a^0 = 1, a^(-n) = 1/a^n.".into(),
            reasoning: "These follow from the definition of exponents as repeated multiplication. They simplify complex expressions.".into(),
            backward_check: "What rules govern operations with powers?".into(),
            related: vec!["powers".into(), "logarithms".into(), "exponential".into()],
            difficulty: 3,
            prerequisites: vec!["multiplication".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "algebra".into(),
            input: "What is a logarithm?".into(),
            output: "log_b(x) = y means b^y = x. The logarithm is the inverse of exponentiation - it answers 'what power gives this result?'".into(),
            reasoning: "log₁₀(100) = 2 because 10² = 100. Logarithms convert multiplication to addition: log(ab) = log(a) + log(b).".into(),
            backward_check: "What is the inverse operation of exponentiation?".into(),
            related: vec!["exponent".into(), "inverse".into(), "exponential growth".into()],
            difficulty: 5,
            prerequisites: vec!["exponents".into()],
        });

        // Calculus
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "calculus".into(),
            input: "What is a derivative?".into(),
            output: "The derivative f'(x) = lim(h→0) [f(x+h) - f(x)]/h measures the instantaneous rate of change of a function. It's the slope of the tangent line.".into(),
            reasoning: "Derivatives give velocity from position, acceleration from velocity. They find maxima/minima where f'(x) = 0.".into(),
            backward_check: "What calculus operation measures instantaneous rate of change?".into(),
            related: vec!["limit".into(), "slope".into(), "integral".into(), "rate".into()],
            difficulty: 5,
            prerequisites: vec!["limits".into(), "functions".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "calculus".into(),
            input: "What is the derivative of x^n?".into(),
            output: "d/dx(x^n) = nx^(n-1). This is the power rule.".into(),
            reasoning: "Applied repeatedly: d/dx(x³) = 3x², d/dx(x²) = 2x, d/dx(x) = 1, d/dx(1) = 0.".into(),
            backward_check: "What rule gives the derivative of a power function?".into(),
            related: vec!["derivative".into(), "power".into(), "polynomial".into()],
            difficulty: 4,
            prerequisites: vec!["derivative definition".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "calculus".into(),
            input: "What is an integral?".into(),
            output: "The integral ∫f(x)dx is the antiderivative - the function whose derivative is f(x). Definite integrals compute area under curves.".into(),
            reasoning: "Integration is the inverse of differentiation. ∫x²dx = x³/3 + C because d/dx(x³/3) = x².".into(),
            backward_check: "What operation is the inverse of differentiation and computes area under curves?".into(),
            related: vec!["derivative".into(), "area".into(), "antiderivative".into()],
            difficulty: 5,
            prerequisites: vec!["derivative".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "calculus".into(),
            input: "What is the Fundamental Theorem of Calculus?".into(),
            output: "If F'(x) = f(x), then ∫[a,b]f(x)dx = F(b) - F(a). Differentiation and integration are inverse operations.".into(),
            reasoning: "This connects the two main operations of calculus. It lets us evaluate definite integrals by finding antiderivatives.".into(),
            backward_check: "What theorem connects differentiation and integration as inverse operations?".into(),
            related: vec!["derivative".into(), "integral".into(), "antiderivative".into()],
            difficulty: 6,
            prerequisites: vec!["derivative".into(), "integral".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "calculus".into(),
            input: "What is the chain rule?".into(),
            output: "d/dx[f(g(x))] = f'(g(x)) · g'(x). For composite functions, multiply the outer derivative by the inner derivative.".into(),
            reasoning: "Example: d/dx(sin(x²)) = cos(x²) · 2x. The chain rule is essential for complex functions.".into(),
            backward_check: "What rule differentiates composite functions?".into(),
            related: vec!["derivative".into(), "composite function".into()],
            difficulty: 5,
            prerequisites: vec!["derivative".into()],
        });

        // Linear Algebra
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "linear_algebra".into(),
            input: "What is a matrix?".into(),
            output: "A matrix is a rectangular array of numbers arranged in rows and columns. An m×n matrix has m rows and n columns.".into(),
            reasoning: "Matrices represent linear transformations, systems of equations, and data. They're fundamental to AI and graphics.".into(),
            backward_check: "What mathematical object is a rectangular array of numbers?".into(),
            related: vec!["vector".into(), "linear transformation".into(), "determinant".into()],
            difficulty: 4,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "linear_algebra".into(),
            input: "How do you multiply matrices?".into(),
            output: "(AB)_ij = Σ_k A_ik × B_kj. Element (i,j) of AB is the dot product of row i of A with column j of B. A must have the same number of columns as B has rows.".into(),
            reasoning: "Matrix multiplication represents composition of linear transformations. It's not commutative: AB ≠ BA in general.".into(),
            backward_check: "What operation computes the dot product of rows and columns to combine matrices?".into(),
            related: vec!["matrix".into(), "dot product".into(), "linear transformation".into()],
            difficulty: 5,
            prerequisites: vec!["matrix".into(), "dot product".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "linear_algebra".into(),
            input: "What is an eigenvalue?".into(),
            output: "For a square matrix A, if Av = λv for some non-zero vector v, then λ is an eigenvalue and v is an eigenvector. The matrix scales v by λ.".into(),
            reasoning: "Eigenvectors point in directions that the transformation only scales (no rotation). Used in PCA, vibration analysis, quantum mechanics.".into(),
            backward_check: "What scalar describes how much a matrix scales its eigenvector?".into(),
            related: vec!["eigenvector".into(), "matrix".into(), "diagonalization".into()],
            difficulty: 7,
            prerequisites: vec!["matrix multiplication".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "linear_algebra".into(),
            input: "What is the dot product?".into(),
            output: "a · b = Σᵢ aᵢbᵢ = |a||b|cos(θ). It's the sum of element-wise products, or the product of magnitudes times cosine of angle between vectors.".into(),
            reasoning: "Dot product measures similarity. Perpendicular vectors have dot product 0. Used in projections, work calculations, and cosine similarity.".into(),
            backward_check: "What operation measures the similarity between two vectors?".into(),
            related: vec!["vector".into(), "cosine similarity".into(), "projection".into()],
            difficulty: 4,
            prerequisites: vec!["vector".into()],
        });

        // Statistics and Probability
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "statistics".into(),
            input: "What is the mean (average)?".into(),
            output: "The mean is the sum of all values divided by the count: μ = (Σxᵢ)/n.".into(),
            reasoning: "The mean is the balance point of a distribution. It's sensitive to outliers unlike the median.".into(),
            backward_check: "What measure of central tendency is the sum divided by count?".into(),
            related: vec!["median".into(), "mode".into(), "variance".into()],
            difficulty: 2,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "statistics".into(),
            input: "What is standard deviation?".into(),
            output: "Standard deviation σ = √[Σ(xᵢ-μ)²/n] measures the spread of data around the mean. About 68% of data falls within ±1σ of the mean.".into(),
            reasoning: "Low σ means data is clustered near the mean; high σ means data is spread out. It's the square root of variance.".into(),
            backward_check: "What measure quantifies the spread of data around the mean?".into(),
            related: vec!["variance".into(), "normal distribution".into(), "mean".into()],
            difficulty: 4,
            prerequisites: vec!["mean".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "probability".into(),
            input: "What is Bayes' Theorem?".into(),
            output: "P(A|B) = P(B|A)P(A) / P(B). It updates probabilities based on new evidence.".into(),
            reasoning: "Given prior belief P(A) and likelihood P(B|A), Bayes' theorem computes posterior P(A|B). Foundation of Bayesian inference.".into(),
            backward_check: "What theorem updates probability estimates based on new evidence?".into(),
            related: vec!["conditional probability".into(), "prior".into(), "posterior".into()],
            difficulty: 6,
            prerequisites: vec!["probability basics".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "probability".into(),
            input: "What is the normal distribution?".into(),
            output: "The normal (Gaussian) distribution has PDF: f(x) = (1/σ√2π)e^(-(x-μ)²/2σ²). It's the bell curve with mean μ and standard deviation σ.".into(),
            reasoning: "Many natural phenomena are normally distributed due to the Central Limit Theorem. 68-95-99.7 rule describes spread.".into(),
            backward_check: "What probability distribution forms a bell-shaped curve?".into(),
            related: vec!["mean".into(), "standard deviation".into(), "Central Limit Theorem".into()],
            difficulty: 5,
            prerequisites: vec!["mean".into(), "standard deviation".into()],
        });

        // Trigonometry
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "trigonometry".into(),
            input: "What are sine, cosine, and tangent?".into(),
            output: "In a right triangle: sin(θ) = opposite/hypotenuse, cos(θ) = adjacent/hypotenuse, tan(θ) = opposite/adjacent = sin(θ)/cos(θ).".into(),
            reasoning: "These ratios depend only on the angle, not triangle size. They extend to the unit circle for all angles.".into(),
            backward_check: "What are the three basic trigonometric ratios in a right triangle?".into(),
            related: vec!["angle".into(), "unit circle".into(), "Pythagorean".into()],
            difficulty: 3,
            prerequisites: vec!["right triangle".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "trigonometry".into(),
            input: "What is the Pythagorean identity?".into(),
            output: "sin²(θ) + cos²(θ) = 1 for all angles θ.".into(),
            reasoning: "From the Pythagorean theorem applied to the unit circle. The point (cos θ, sin θ) lies on the unit circle.".into(),
            backward_check: "What identity relates sine squared and cosine squared?".into(),
            related: vec!["sine".into(), "cosine".into(), "unit circle".into()],
            difficulty: 3,
            prerequisites: vec!["sin cos tan".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Mathematics,
            subcategory: "trigonometry".into(),
            input: "What is Euler's formula?".into(),
            output: "e^(iθ) = cos(θ) + i·sin(θ). This connects exponentials with trigonometry via complex numbers.".into(),
            reasoning: "Setting θ = π gives Euler's identity: e^(iπ) + 1 = 0, connecting five fundamental constants.".into(),
            backward_check: "What formula expresses complex exponentials in terms of sine and cosine?".into(),
            related: vec!["complex numbers".into(), "exponential".into(), "trigonometry".into()],
            difficulty: 7,
            prerequisites: vec!["complex numbers".into(), "trigonometry".into()],
        });
    }

    // =========================================================================
    // LANGUAGE KNOWLEDGE
    // =========================================================================
    fn load_language(&mut self) {
        // Grammar
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "grammar".into(),
            input: "What are the parts of speech?".into(),
            output: "Noun (person/place/thing), Verb (action/state), Adjective (describes noun), Adverb (describes verb/adj), Pronoun (replaces noun), Preposition (shows relationship), Conjunction (connects words/clauses), Interjection (expresses emotion).".into(),
            reasoning: "Parts of speech categorize words by their grammatical function. Understanding them is essential for proper sentence construction.".into(),
            backward_check: "What are the grammatical categories that classify all words?".into(),
            related: vec!["syntax".into(), "sentence".into(), "clause".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "grammar".into(),
            input: "What is a sentence?".into(),
            output: "A sentence is a grammatical unit containing at least a subject and predicate, expressing a complete thought. It begins with a capital letter and ends with punctuation.".into(),
            reasoning: "Complete sentences have who/what (subject) doing/being something (predicate). Fragments lack one or both.".into(),
            backward_check: "What grammatical unit expresses a complete thought with subject and predicate?".into(),
            related: vec!["subject".into(), "predicate".into(), "clause".into()],
            difficulty: 2,
            prerequisites: vec!["parts of speech".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "grammar".into(),
            input: "What are verb tenses?".into(),
            output: "Past (happened before), Present (happening now), Future (will happen). Each has simple, continuous, perfect, and perfect continuous forms.".into(),
            reasoning: "Tenses locate actions in time. 'I walk' (present), 'I walked' (past), 'I will walk' (future), 'I have walked' (present perfect).".into(),
            backward_check: "What grammatical feature indicates when an action occurs?".into(),
            related: vec!["verb".into(), "time".into(), "aspect".into()],
            difficulty: 4,
            prerequisites: vec!["verb".into()],
        });

        // Semantics
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "semantics".into(),
            input: "What is a synonym?".into(),
            output: "A synonym is a word with the same or similar meaning as another word. Example: 'happy' and 'joyful' are synonyms.".into(),
            reasoning: "Synonyms allow varied expression and avoid repetition. Context matters - 'big' and 'large' are synonyms but 'big sister' ≠ 'large sister'.".into(),
            backward_check: "What term describes words with similar meanings?".into(),
            related: vec!["antonym".into(), "vocabulary".into(), "meaning".into()],
            difficulty: 2,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "semantics".into(),
            input: "What is an antonym?".into(),
            output: "An antonym is a word with opposite meaning. Example: 'hot' and 'cold', 'up' and 'down' are antonyms.".into(),
            reasoning: "Antonyms define concepts by contrast. Some are gradable (hot-cold has degrees), others are complementary (alive-dead).".into(),
            backward_check: "What term describes words with opposite meanings?".into(),
            related: vec!["synonym".into(), "opposite".into(), "meaning".into()],
            difficulty: 2,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "semantics".into(),
            input: "What is context in language?".into(),
            output: "Context is the surrounding information that gives meaning to words and sentences. It includes situational context (physical setting), linguistic context (surrounding words), and cultural context (shared knowledge).".into(),
            reasoning: "'Bank' means different things in 'river bank' vs 'savings bank'. Context disambiguates meaning.".into(),
            backward_check: "What provides the surrounding information needed to interpret meaning?".into(),
            related: vec!["ambiguity".into(), "pragmatics".into(), "meaning".into()],
            difficulty: 4,
            prerequisites: vec![],
        });

        // Syntax
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "syntax".into(),
            input: "What is syntax?".into(),
            output: "Syntax is the set of rules governing how words combine to form phrases, clauses, and sentences. It determines word order and grammatical relationships.".into(),
            reasoning: "In English, syntax typically follows Subject-Verb-Object order. 'Dog bites man' ≠ 'Man bites dog' - word order changes meaning.".into(),
            backward_check: "What linguistic system governs the arrangement of words into sentences?".into(),
            related: vec!["grammar".into(), "word order".into(), "phrase structure".into()],
            difficulty: 5,
            prerequisites: vec!["parts of speech".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "syntax".into(),
            input: "What is a clause?".into(),
            output: "A clause is a group of words containing a subject and verb. Independent clauses can stand alone; dependent clauses cannot and need an independent clause.".into(),
            reasoning: "'When I arrive' is dependent (incomplete thought). 'I will call you' is independent. Combined: 'When I arrive, I will call you.'".into(),
            backward_check: "What grammatical unit contains a subject and verb and forms part of a sentence?".into(),
            related: vec!["sentence".into(), "phrase".into(), "conjunction".into()],
            difficulty: 4,
            prerequisites: vec!["sentence".into()],
        });

        // Rhetoric and Communication
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "rhetoric".into(),
            input: "What is a metaphor?".into(),
            output: "A metaphor is a figure of speech that describes something by saying it IS something else. 'Time is money' compares time to money directly.".into(),
            reasoning: "Unlike similes (uses 'like' or 'as'), metaphors make direct comparisons. They create vivid imagery and deeper meaning.".into(),
            backward_check: "What figure of speech makes a direct comparison without 'like' or 'as'?".into(),
            related: vec!["simile".into(), "analogy".into(), "figurative language".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Language,
            subcategory: "rhetoric".into(),
            input: "What is an analogy?".into(),
            output: "An analogy explains something by comparing it to something similar. 'The brain is like a computer' uses a familiar concept to explain an unfamiliar one.".into(),
            reasoning: "Analogies aid understanding by mapping relationships. A:B :: C:D means A relates to B as C relates to D.".into(),
            backward_check: "What comparison technique explains unfamiliar concepts using familiar ones?".into(),
            related: vec!["metaphor".into(), "simile".into(), "comparison".into()],
            difficulty: 3,
            prerequisites: vec![],
        });
    }

    // =========================================================================
    // COMPUTER SCIENCE KNOWLEDGE
    // =========================================================================
    fn load_computer_science(&mut self) {
        // Algorithms
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "algorithms".into(),
            input: "What is Big O notation?".into(),
            output: "Big O describes algorithm time/space complexity as input grows. O(1) constant, O(log n) logarithmic, O(n) linear, O(n log n) linearithmic, O(n²) quadratic, O(2ⁿ) exponential.".into(),
            reasoning: "Big O shows scalability. Binary search O(log n) beats linear search O(n) for large n. Bubble sort O(n²) is slow; merge sort O(n log n) is faster.".into(),
            backward_check: "What notation describes how algorithm performance scales with input size?".into(),
            related: vec!["complexity".into(), "algorithm".into(), "performance".into()],
            difficulty: 5,
            prerequisites: vec!["algorithms basics".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "algorithms".into(),
            input: "What is binary search?".into(),
            output: "Binary search finds a target in a sorted array by repeatedly halving the search space. Compare target to middle element; if target is smaller, search left half; if larger, search right half. O(log n) complexity.".into(),
            reasoning: "Each comparison eliminates half the remaining elements. For 1 million elements, at most 20 comparisons needed (log₂(10⁶) ≈ 20).".into(),
            backward_check: "What search algorithm divides the search space in half each step?".into(),
            related: vec!["sorting".into(), "searching".into(), "divide and conquer".into()],
            difficulty: 4,
            prerequisites: vec!["arrays".into(), "sorting".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "algorithms".into(),
            input: "What is recursion?".into(),
            output: "Recursion is when a function calls itself to solve smaller subproblems. It requires: 1) Base case (termination condition), 2) Recursive case (self-call with smaller input).".into(),
            reasoning: "Factorial: n! = n × (n-1)!. Base: 0! = 1. Recursive: 5! = 5 × 4! = 5 × 4 × 3! = ... = 120.".into(),
            backward_check: "What programming technique involves a function calling itself?".into(),
            related: vec!["iteration".into(), "stack".into(), "divide and conquer".into()],
            difficulty: 5,
            prerequisites: vec!["functions".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "algorithms".into(),
            input: "What is dynamic programming?".into(),
            output: "Dynamic programming solves complex problems by breaking them into overlapping subproblems, solving each once, and storing results to avoid recomputation. Uses memoization or tabulation.".into(),
            reasoning: "Fibonacci naively is O(2ⁿ). With DP, it's O(n) by storing previously computed values. Used in shortest paths, sequence alignment, optimization.".into(),
            backward_check: "What optimization technique stores subproblem solutions to avoid recomputation?".into(),
            related: vec!["memoization".into(), "optimization".into(), "recursion".into()],
            difficulty: 7,
            prerequisites: vec!["recursion".into()],
        });

        // Data Structures
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "data_structures".into(),
            input: "What is an array?".into(),
            output: "An array is a contiguous block of memory storing elements of the same type. Access by index is O(1). Insert/delete at arbitrary position is O(n).".into(),
            reasoning: "Arrays provide fast random access but slow insertion. Memory address = base + index × element_size.".into(),
            backward_check: "What data structure stores elements contiguously with O(1) index access?".into(),
            related: vec!["list".into(), "memory".into(), "indexing".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "data_structures".into(),
            input: "What is a hash table?".into(),
            output: "A hash table maps keys to values using a hash function. Average case: O(1) insert, delete, lookup. Collisions handled by chaining or open addressing.".into(),
            reasoning: "Hash function converts key to array index. Good hash functions distribute keys uniformly. Used in dictionaries, caches, databases.".into(),
            backward_check: "What data structure provides O(1) average case key-value lookup?".into(),
            related: vec!["hash function".into(), "dictionary".into(), "collision".into()],
            difficulty: 5,
            prerequisites: vec!["array".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "data_structures".into(),
            input: "What is a binary tree?".into(),
            output: "A binary tree is a hierarchical structure where each node has at most two children (left and right). Binary Search Trees maintain ordering: left < parent < right.".into(),
            reasoning: "BST operations are O(log n) average, O(n) worst case (unbalanced). Balanced trees (AVL, Red-Black) guarantee O(log n).".into(),
            backward_check: "What tree structure has at most two children per node?".into(),
            related: vec!["BST".into(), "tree traversal".into(), "balanced tree".into()],
            difficulty: 5,
            prerequisites: vec!["recursion".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "data_structures".into(),
            input: "What is a graph?".into(),
            output: "A graph G = (V, E) consists of vertices V and edges E connecting them. Directed graphs have one-way edges; undirected have two-way. Weighted graphs have edge costs.".into(),
            reasoning: "Represented as adjacency matrix O(V²) space or adjacency list O(V+E) space. Used for networks, maps, relationships.".into(),
            backward_check: "What data structure represents objects and their pairwise connections?".into(),
            related: vec!["BFS".into(), "DFS".into(), "shortest path".into()],
            difficulty: 6,
            prerequisites: vec!["arrays".into(), "linked lists".into()],
        });

        // Machine Learning
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "machine_learning".into(),
            input: "What is gradient descent?".into(),
            output: "Gradient descent minimizes a function by iteratively moving in the direction of steepest descent (negative gradient). Update: θ = θ - α∇J(θ), where α is learning rate.".into(),
            reasoning: "Like rolling downhill to find the lowest point. Small α is slow but stable; large α may overshoot. Variants: SGD, Adam, RMSprop.".into(),
            backward_check: "What optimization algorithm iteratively moves toward the minimum of a function?".into(),
            related: vec!["optimization".into(), "neural network".into(), "loss function".into()],
            difficulty: 6,
            prerequisites: vec!["calculus".into(), "linear algebra".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "machine_learning".into(),
            input: "What is backpropagation?".into(),
            output: "Backpropagation computes gradients of the loss function with respect to network weights using the chain rule, propagating errors backward from output to input layers.".into(),
            reasoning: "Forward pass computes outputs. Backward pass computes gradients. Weights updated to minimize loss. Enables training deep networks.".into(),
            backward_check: "What algorithm computes gradients in neural networks using the chain rule?".into(),
            related: vec!["gradient descent".into(), "neural network".into(), "chain rule".into()],
            difficulty: 7,
            prerequisites: vec!["gradient descent".into(), "chain rule".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "machine_learning".into(),
            input: "What is a neural network?".into(),
            output: "A neural network is layers of interconnected nodes (neurons). Each connection has a weight. Neurons apply activation functions to weighted sums of inputs.".into(),
            reasoning: "Input layer receives data, hidden layers learn features, output layer produces predictions. Universal approximation theorem: can approximate any continuous function.".into(),
            backward_check: "What computational model consists of layers of connected artificial neurons?".into(),
            related: vec!["activation function".into(), "backpropagation".into(), "deep learning".into()],
            difficulty: 6,
            prerequisites: vec!["linear algebra".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::ComputerScience,
            subcategory: "machine_learning".into(),
            input: "What is attention in neural networks?".into(),
            output: "Attention allows models to focus on relevant parts of input. Attention(Q,K,V) = softmax(QK^T/√d_k)V. Queries attend to keys to weight values.".into(),
            reasoning: "Instead of fixed-size context, attention dynamically weighs all input positions. Enables Transformers to process sequences effectively.".into(),
            backward_check: "What mechanism allows neural networks to focus on relevant input parts?".into(),
            related: vec!["transformer".into(), "self-attention".into(), "sequence modeling".into()],
            difficulty: 8,
            prerequisites: vec!["neural networks".into(), "softmax".into()],
        });
    }

    // =========================================================================
    // CHEMISTRY KNOWLEDGE
    // =========================================================================
    fn load_chemistry(&mut self) {
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Chemistry,
            subcategory: "atomic_structure".into(),
            input: "What is an atom?".into(),
            output: "An atom is the smallest unit of matter that retains the properties of an element. It consists of a nucleus (protons and neutrons) surrounded by electrons.".into(),
            reasoning: "Protons determine element (atomic number). Electrons determine chemical behavior. Neutrons add mass but don't change element.".into(),
            backward_check: "What is the smallest unit of an element that retains its chemical properties?".into(),
            related: vec!["proton".into(), "electron".into(), "neutron".into(), "element".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Chemistry,
            subcategory: "periodic_table".into(),
            input: "What is the periodic table?".into(),
            output: "The periodic table organizes elements by atomic number (protons) into periods (rows) and groups (columns). Elements in the same group have similar properties due to similar electron configurations.".into(),
            reasoning: "Mendeleev predicted missing elements. Periodic trends: atomic radius decreases across periods, increases down groups. Electronegativity increases across periods.".into(),
            backward_check: "What chart organizes elements by atomic number showing periodic patterns?".into(),
            related: vec!["elements".into(), "periods".into(), "groups".into(), "electrons".into()],
            difficulty: 4,
            prerequisites: vec!["atoms".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Chemistry,
            subcategory: "bonding".into(),
            input: "What is a covalent bond?".into(),
            output: "A covalent bond forms when atoms share electrons to achieve stable electron configurations. Common in molecules like H₂O, CO₂, and organic compounds.".into(),
            reasoning: "Atoms share to fill outer shells. Single bonds share 2 electrons, double bonds 4, triple bonds 6. Bond strength increases with bond order.".into(),
            backward_check: "What type of chemical bond involves sharing electrons between atoms?".into(),
            related: vec!["ionic bond".into(), "electrons".into(), "molecule".into()],
            difficulty: 4,
            prerequisites: vec!["atomic structure".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Chemistry,
            subcategory: "reactions".into(),
            input: "What is a chemical equation?".into(),
            output: "A chemical equation shows reactants transforming into products. It must be balanced: same number of each atom on both sides (conservation of mass).".into(),
            reasoning: "2H₂ + O₂ → 2H₂O shows 4 H and 2 O on each side. Coefficients indicate relative amounts (molar ratios).".into(),
            backward_check: "What notation represents a chemical reaction with balanced reactants and products?".into(),
            related: vec!["reactants".into(), "products".into(), "stoichiometry".into()],
            difficulty: 4,
            prerequisites: vec!["chemical formulas".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Chemistry,
            subcategory: "acids_bases".into(),
            input: "What is pH?".into(),
            output: "pH = -log₁₀[H⁺] measures acidity. pH 7 is neutral, <7 is acidic, >7 is basic. Each pH unit represents a 10× change in H⁺ concentration.".into(),
            reasoning: "Pure water: [H⁺] = 10⁻⁷ M, so pH = 7. Stomach acid pH ≈ 1 (very acidic). Blood pH ≈ 7.4 (slightly basic).".into(),
            backward_check: "What scale measures the acidity or basicity of a solution?".into(),
            related: vec!["acid".into(), "base".into(), "hydrogen ion".into()],
            difficulty: 5,
            prerequisites: vec!["logarithms".into(), "acids".into()],
        });
    }

    // =========================================================================
    // BIOLOGY KNOWLEDGE
    // =========================================================================
    fn load_biology(&mut self) {
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Biology,
            subcategory: "cell_biology".into(),
            input: "What is a cell?".into(),
            output: "A cell is the basic structural and functional unit of all living organisms. Prokaryotic cells lack a nucleus; eukaryotic cells have membrane-bound organelles including a nucleus.".into(),
            reasoning: "Cells carry out life processes: metabolism, growth, reproduction. All cells come from pre-existing cells (cell theory).".into(),
            backward_check: "What is the fundamental unit of life that all organisms are made of?".into(),
            related: vec!["organism".into(), "nucleus".into(), "organelles".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Biology,
            subcategory: "genetics".into(),
            input: "What is DNA?".into(),
            output: "DNA (deoxyribonucleic acid) is a double helix molecule storing genetic information. It consists of nucleotides with bases A, T, G, C. A pairs with T, G pairs with C.".into(),
            reasoning: "DNA sequence encodes proteins via transcription (DNA→RNA) and translation (RNA→protein). The genetic code is nearly universal across life.".into(),
            backward_check: "What molecule stores genetic information in a double helix structure?".into(),
            related: vec!["RNA".into(), "genes".into(), "chromosomes".into(), "heredity".into()],
            difficulty: 5,
            prerequisites: vec!["cell".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Biology,
            subcategory: "evolution".into(),
            input: "What is natural selection?".into(),
            output: "Natural selection is the process where organisms with traits better suited to their environment tend to survive and reproduce more, passing those traits to offspring.".into(),
            reasoning: "Variation exists. Some variants have higher fitness. Traits are heritable. Over generations, beneficial traits become more common. This drives evolution.".into(),
            backward_check: "What evolutionary mechanism causes beneficial traits to become more common over time?".into(),
            related: vec!["evolution".into(), "fitness".into(), "adaptation".into(), "Darwin".into()],
            difficulty: 5,
            prerequisites: vec!["genetics basics".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Biology,
            subcategory: "ecology".into(),
            input: "What is an ecosystem?".into(),
            output: "An ecosystem is a community of living organisms interacting with each other and their physical environment. It includes producers, consumers, decomposers, and abiotic factors.".into(),
            reasoning: "Energy flows from sun → producers → consumers. Nutrients cycle through food webs. Ecosystems can be small (pond) or large (rainforest).".into(),
            backward_check: "What system includes living organisms and their physical environment interacting together?".into(),
            related: vec!["food web".into(), "biodiversity".into(), "habitat".into()],
            difficulty: 4,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Biology,
            subcategory: "physiology".into(),
            input: "How does photosynthesis work?".into(),
            output: "Photosynthesis converts light energy, CO₂, and H₂O into glucose and O₂: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. Occurs in chloroplasts.".into(),
            reasoning: "Light reactions in thylakoids produce ATP and NADPH. Dark reactions (Calvin cycle) use these to fix CO₂ into sugar.".into(),
            backward_check: "What process do plants use to convert sunlight into chemical energy?".into(),
            related: vec!["chlorophyll".into(), "cellular respiration".into(), "glucose".into()],
            difficulty: 6,
            prerequisites: vec!["cell".into(), "energy".into()],
        });
    }

    // =========================================================================
    // LOGIC KNOWLEDGE
    // =========================================================================
    fn load_logic(&mut self) {
        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Logic,
            subcategory: "propositional".into(),
            input: "What is a logical proposition?".into(),
            output: "A proposition is a declarative statement that is either true or false, but not both. 'It is raining' is a proposition. Questions and commands are not propositions.".into(),
            reasoning: "Propositions are the basic units of logical reasoning. They can be combined with operators (AND, OR, NOT, IF-THEN) to form complex statements.".into(),
            backward_check: "What is a statement that can be assigned a truth value?".into(),
            related: vec!["truth value".into(), "logic".into(), "argument".into()],
            difficulty: 3,
            prerequisites: vec![],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Logic,
            subcategory: "propositional".into(),
            input: "What is modus ponens?".into(),
            output: "Modus ponens: If P then Q. P is true. Therefore Q is true. Symbolically: ((P → Q) ∧ P) → Q.".into(),
            reasoning: "Example: If it rains, the ground is wet. It is raining. Therefore, the ground is wet. This is a valid form of deductive reasoning.".into(),
            backward_check: "What logical rule infers the consequent when the antecedent is true?".into(),
            related: vec!["implication".into(), "deduction".into(), "modus tollens".into()],
            difficulty: 4,
            prerequisites: vec!["proposition".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Logic,
            subcategory: "propositional".into(),
            input: "What is modus tollens?".into(),
            output: "Modus tollens: If P then Q. Q is false. Therefore P is false. Symbolically: ((P → Q) ∧ ¬Q) → ¬P.".into(),
            reasoning: "Example: If it rained, the ground is wet. The ground is not wet. Therefore, it did not rain. This is denying the consequent.".into(),
            backward_check: "What logical rule infers the negation of the antecedent when the consequent is false?".into(),
            related: vec!["implication".into(), "contrapositive".into(), "modus ponens".into()],
            difficulty: 4,
            prerequisites: vec!["modus ponens".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Logic,
            subcategory: "fallacies".into(),
            input: "What is a logical fallacy?".into(),
            output: "A logical fallacy is an error in reasoning that undermines the logic of an argument. Common types: ad hominem (attack the person), straw man (misrepresent argument), false dichotomy (only two options when more exist).".into(),
            reasoning: "Fallacies may be persuasive but are logically invalid. Recognizing fallacies is crucial for critical thinking and evaluating arguments.".into(),
            backward_check: "What term describes errors in reasoning that invalidate arguments?".into(),
            related: vec!["argument".into(), "reasoning".into(), "critical thinking".into()],
            difficulty: 5,
            prerequisites: vec!["valid argument".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Logic,
            subcategory: "predicate".into(),
            input: "What is predicate logic?".into(),
            output: "Predicate logic extends propositional logic with quantifiers (∀ = for all, ∃ = there exists) and predicates (properties/relations). ∀x P(x) means 'for all x, P(x) is true'.".into(),
            reasoning: "Predicate logic can express 'All humans are mortal' as ∀x(Human(x) → Mortal(x)). More expressive than propositional logic.".into(),
            backward_check: "What extension of propositional logic uses quantifiers and predicates?".into(),
            related: vec!["quantifier".into(), "first-order logic".into(), "proposition".into()],
            difficulty: 6,
            prerequisites: vec!["propositional logic".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Logic,
            subcategory: "proof".into(),
            input: "What is proof by contradiction?".into(),
            output: "To prove P, assume ¬P and derive a contradiction. Since the assumption leads to impossibility, P must be true.".into(),
            reasoning: "Example: To prove √2 is irrational, assume it's rational (a/b in lowest terms), derive that both a and b are even (contradiction). Therefore √2 is irrational.".into(),
            backward_check: "What proof technique assumes the negation and derives a contradiction?".into(),
            related: vec!["proof".into(), "negation".into(), "reductio ad absurdum".into()],
            difficulty: 6,
            prerequisites: vec!["logical reasoning".into()],
        });

        self.add_item(KnowledgeItem {
            category: KnowledgeCategory::Logic,
            subcategory: "proof".into(),
            input: "What is mathematical induction?".into(),
            output: "Mathematical induction proves statements for all natural numbers: 1) Base case: Prove P(1). 2) Inductive step: Prove P(k) → P(k+1). Then P(n) is true for all n ≥ 1.".into(),
            reasoning: "Like dominoes: first falls (base), each knocks next (inductive step), so all fall. Proves infinite statements with finite proof.".into(),
            backward_check: "What proof technique proves statements for all natural numbers using base case and inductive step?".into(),
            related: vec!["proof".into(), "recursion".into(), "natural numbers".into()],
            difficulty: 6,
            prerequisites: vec!["proof".into()],
        });
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_base_creation() {
        let kb = KnowledgeBase::new();
        assert!(kb.len() > 50);
    }

    #[test]
    fn test_category_filtering() {
        let kb = KnowledgeBase::new();
        let physics = kb.by_category(KnowledgeCategory::Physics);
        assert!(!physics.is_empty());
        assert!(physics.iter().all(|item| item.category == KnowledgeCategory::Physics));
    }

    #[test]
    fn test_backward_check_exists() {
        let kb = KnowledgeBase::new();
        for item in kb.all_items() {
            assert!(!item.backward_check.is_empty(), "Item missing backward check: {}", item.input);
        }
    }

    #[test]
    fn test_reasoning_exists() {
        let kb = KnowledgeBase::new();
        for item in kb.all_items() {
            assert!(!item.reasoning.is_empty(), "Item missing reasoning: {}", item.input);
        }
    }
}
