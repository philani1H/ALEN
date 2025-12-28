#!/bin/bash
# ALEN - Comprehensive Training and Testing Script
# This script trains ALEN on all knowledge categories and tests its understanding

set -e

BASE_URL="${BASE_URL:-http://localhost:3000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║     █████╗ ██╗     ███████╗███╗   ██╗                                    ║"
echo "║    ██╔══██╗██║     ██╔════╝████╗  ██║                                    ║"
echo "║    ███████║██║     █████╗  ██╔██╗ ██║                                    ║"
echo "║    ██╔══██║██║     ██╔══╝  ██║╚██╗██║                                    ║"
echo "║    ██║  ██║███████╗███████╗██║ ╚████║                                    ║"
echo "║    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝                                    ║"
echo "║                                                                           ║"
echo "║          Advanced Learning Engine with Neural verification                ║"
echo "║                    Comprehensive Training Suite                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to display section headers
section() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
}

# Function to train a single item
train_item() {
    local input="$1"
    local expected="$2"
    local context="${3:-}"
    
    local json
    if [ -n "$context" ]; then
        json=$(jq -n --arg i "$input" --arg e "$expected" --arg c "$context" \
            '{input: $i, expected_answer: $e, context: $c}')
    else
        json=$(jq -n --arg i "$input" --arg e "$expected" \
            '{input: $i, expected_answer: $e}')
    fi
    
    result=$(curl -s -X POST "$BASE_URL/train" \
        -H "Content-Type: application/json" \
        -d "$json")
    
    success=$(echo "$result" | jq -r '.success')
    verified=$(echo "$result" | jq -r '.verified')
    confidence=$(echo "$result" | jq -r '.confidence')
    
    if [ "$verified" == "true" ]; then
        echo -e "${GREEN}✓${NC} Verified: $input → confidence: ${confidence}"
    else
        echo -e "${YELLOW}○${NC} Learning: $input → confidence: ${confidence}"
    fi
}

# Function to test understanding
test_understanding() {
    local input="$1"
    
    result=$(curl -s -X POST "$BASE_URL/infer" \
        -H "Content-Type: application/json" \
        -d "{\"input\": \"$input\"}")
    
    confidence=$(echo "$result" | jq -r '.confidence')
    iterations=$(echo "$result" | jq -r '.iterations')
    
    echo -e "  Query: ${BLUE}$input${NC}"
    echo -e "  Confidence: $confidence, Iterations: $iterations"
}

# Check health first
section "System Health Check"
health=$(curl -s "$BASE_URL/health" 2>/dev/null || echo '{"status":"error"}')
status=$(echo "$health" | jq -r '.status' 2>/dev/null || echo "error")

if [ "$status" == "ok" ]; then
    echo -e "${GREEN}✓ ALEN is running and healthy${NC}"
else
    echo -e "${RED}✗ ALEN is not responding at $BASE_URL${NC}"
    echo -e "${YELLOW}Please start ALEN first with: cargo run${NC}"
    exit 1
fi

# ============================================================================
# MATHEMATICS TRAINING
# ============================================================================
section "Phase 1: Mathematics Foundations"

echo -e "${YELLOW}Training arithmetic...${NC}"
train_item "What is 2 + 2?" "4" "basic arithmetic addition"
train_item "What is 7 × 8?" "56" "multiplication"
train_item "What is 100 / 4?" "25" "division"
train_item "What is 15 - 9?" "6" "subtraction"
train_item "What is 3^4?" "81" "exponentiation"

echo -e "\n${YELLOW}Training algebra...${NC}"
train_item "Solve for x: 2x + 5 = 13" "x = 4" "linear equations"
train_item "What is the quadratic formula?" "x = (-b ± √(b²-4ac)) / 2a" "quadratic equations"
train_item "Factor x² - 9" "(x+3)(x-3)" "difference of squares"
train_item "Simplify (x²)³" "x⁶" "exponent rules"

echo -e "\n${YELLOW}Training calculus...${NC}"
train_item "What is the derivative of x²?" "2x" "power rule differentiation"
train_item "What is the derivative of sin(x)?" "cos(x)" "trigonometric derivatives"
train_item "What is the integral of 2x?" "x² + C" "basic integration"
train_item "What is the chain rule?" "d/dx[f(g(x))] = f'(g(x)) · g'(x)" "composite function differentiation"

echo -e "\n${YELLOW}Training linear algebra...${NC}"
train_item "What is a matrix?" "A rectangular array of numbers arranged in rows and columns" "linear algebra basics"
train_item "What is the dot product of vectors?" "Sum of element-wise products" "vector operations"
train_item "What is an eigenvalue?" "λ where Av = λv for some vector v" "matrix eigenvalues"

# ============================================================================
# PHYSICS TRAINING
# ============================================================================
section "Phase 2: Physics Fundamentals"

echo -e "${YELLOW}Training classical mechanics...${NC}"
train_item "What is Newton's First Law?" "An object at rest stays at rest unless acted upon by a force" "inertia"
train_item "What is F = ma?" "Force equals mass times acceleration" "Newton's second law"
train_item "What is kinetic energy?" "KE = ½mv²" "energy of motion"
train_item "What is momentum?" "p = mv, the product of mass and velocity" "conservation laws"
train_item "What is the law of conservation of energy?" "Energy cannot be created or destroyed, only transformed" "thermodynamics"

echo -e "\n${YELLOW}Training thermodynamics...${NC}"
train_item "What is the First Law of Thermodynamics?" "ΔU = Q - W, energy is conserved" "heat and work"
train_item "What is entropy?" "A measure of disorder or randomness in a system" "second law"
train_item "What is absolute zero?" "0 Kelvin, the lowest possible temperature" "temperature"

echo -e "\n${YELLOW}Training electromagnetism...${NC}"
train_item "What is Coulomb's Law?" "F = kq₁q₂/r², force between charges" "electrostatics"
train_item "What is Ohm's Law?" "V = IR, voltage equals current times resistance" "circuits"
train_item "What is electromagnetic induction?" "Changing magnetic fields create electric fields" "Faraday's law"

echo -e "\n${YELLOW}Training quantum mechanics...${NC}"
train_item "What is the Heisenberg Uncertainty Principle?" "Cannot know both position and momentum precisely" "quantum uncertainty"
train_item "What is wave-particle duality?" "Matter exhibits both wave and particle properties" "quantum nature"
train_item "What is Schrödinger's equation?" "iℏ∂ψ/∂t = Ĥψ, describes quantum state evolution" "wave function"

echo -e "\n${YELLOW}Training relativity...${NC}"
train_item "What is E=mc²?" "Energy equals mass times speed of light squared" "mass-energy equivalence"
train_item "What is time dilation?" "Time passes slower at high speeds or in gravity" "special relativity"
train_item "What is general relativity?" "Gravity is the curvature of spacetime" "Einstein's theory"

# ============================================================================
# COMPUTER SCIENCE TRAINING
# ============================================================================
section "Phase 3: Computer Science"

echo -e "${YELLOW}Training algorithms...${NC}"
train_item "What is Big O notation?" "Describes algorithm time/space complexity" "complexity analysis"
train_item "What is binary search?" "O(log n) search by halving the search space" "divide and conquer"
train_item "What is recursion?" "A function that calls itself" "self-reference"
train_item "What is dynamic programming?" "Solving by storing subproblem solutions" "optimization"

echo -e "\n${YELLOW}Training data structures...${NC}"
train_item "What is an array?" "Contiguous memory with O(1) index access" "sequential data"
train_item "What is a hash table?" "O(1) average key-value lookup using hash function" "associative data"
train_item "What is a binary tree?" "Hierarchical structure with at most two children per node" "tree structures"
train_item "What is a graph?" "Vertices connected by edges" "network structures"

echo -e "\n${YELLOW}Training machine learning...${NC}"
train_item "What is gradient descent?" "Minimize function by moving in direction of steepest descent" "optimization"
train_item "What is backpropagation?" "Compute gradients using chain rule through neural network" "deep learning"
train_item "What is attention in neural networks?" "Mechanism to focus on relevant parts of input" "transformers"

# ============================================================================
# LANGUAGE TRAINING
# ============================================================================
section "Phase 4: Language Understanding"

echo -e "${YELLOW}Training grammar...${NC}"
train_item "What are the parts of speech?" "Noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection" "grammar basics"
train_item "What is a sentence?" "A grammatical unit with subject and predicate expressing complete thought" "syntax"
train_item "What are verb tenses?" "Past, present, future - indicating when action occurs" "temporal grammar"

echo -e "\n${YELLOW}Training semantics...${NC}"
train_item "What is a synonym?" "A word with the same or similar meaning" "vocabulary"
train_item "What is an antonym?" "A word with opposite meaning" "vocabulary"
train_item "What is context in language?" "Surrounding information that gives meaning" "pragmatics"

echo -e "\n${YELLOW}Training rhetoric...${NC}"
train_item "What is a metaphor?" "Direct comparison saying something IS something else" "figurative language"
train_item "What is an analogy?" "Explaining by comparing to something similar" "reasoning"

# ============================================================================
# LOGIC AND REASONING TRAINING
# ============================================================================
section "Phase 5: Logic and Reasoning"

echo -e "${YELLOW}Training formal logic...${NC}"
train_item "What is modus ponens?" "If P then Q; P is true; therefore Q" "deductive reasoning"
train_item "What is modus tollens?" "If P then Q; Q is false; therefore P is false" "deductive reasoning"
train_item "What is a logical fallacy?" "An error in reasoning that invalidates an argument" "critical thinking"
train_item "What is proof by contradiction?" "Assume negation, derive contradiction, conclude original true" "proof techniques"

# ============================================================================
# CHEMISTRY AND BIOLOGY TRAINING
# ============================================================================
section "Phase 6: Natural Sciences"

echo -e "${YELLOW}Training chemistry...${NC}"
train_item "What is an atom?" "Smallest unit of matter retaining element properties" "atomic structure"
train_item "What is a covalent bond?" "Atoms sharing electrons" "chemical bonding"
train_item "What is pH?" "Measure of acidity, pH = -log[H+]" "acids and bases"

echo -e "\n${YELLOW}Training biology...${NC}"
train_item "What is a cell?" "Basic structural and functional unit of life" "cell biology"
train_item "What is DNA?" "Double helix molecule storing genetic information" "genetics"
train_item "What is natural selection?" "Organisms with beneficial traits survive and reproduce more" "evolution"
train_item "What is photosynthesis?" "Converting light, CO2, and H2O into glucose and O2" "plant biology"

# ============================================================================
# TESTING UNDERSTANDING
# ============================================================================
section "Phase 7: Testing Understanding (Verification)"

echo -e "${YELLOW}Testing with verification - backward inference check...${NC}"
echo -e "${BLUE}ALEN must prove it understands by working backward from answers${NC}"
echo ""

echo -e "${CYAN}Mathematics Tests:${NC}"
test_understanding "If the answer is 4, what simple addition problem gives this?"
test_understanding "What calculus concept gives 2x as its result?"
test_understanding "If det(A) = 0, what does this tell us about matrix A?"

echo ""
echo -e "${CYAN}Physics Tests:${NC}"
test_understanding "If acceleration is zero, what can we say about forces?"
test_understanding "If energy is conserved, what cannot happen?"
test_understanding "If time dilates, what must be happening?"

echo ""
echo -e "${CYAN}Logic Tests:${NC}"
test_understanding "If we know Q is false and P implies Q, what must be true?"
test_understanding "Why is correlation not causation?"

# ============================================================================
# FINAL STATISTICS
# ============================================================================
section "Training Complete - Statistics"

stats=$(curl -s "$BASE_URL/stats")
echo -e "${GREEN}Final Statistics:${NC}"
echo "$stats" | jq '.'

echo ""
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}ALEN Training Complete!${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "The system has been trained on:"
echo "  • Mathematics (arithmetic, algebra, calculus, linear algebra)"
echo "  • Physics (mechanics, thermodynamics, E&M, quantum, relativity)"
echo "  • Computer Science (algorithms, data structures, ML)"
echo "  • Language (grammar, semantics, rhetoric)"
echo "  • Logic and Reasoning (formal logic, proofs)"
echo "  • Natural Sciences (chemistry, biology)"
echo ""
echo "ALEN uses verification-first learning:"
echo "  1. Forward check: Does solution match expected?"
echo "  2. Backward check: Can we reconstruct problem from solution?"
echo "  3. Confidence check: Is the model confident?"
echo "  4. Energy check: Is this a low-energy (stable) solution?"
echo "  5. Coherence check: Does this align with existing knowledge?"
echo ""
echo "Only when ALL checks pass does learning commit to memory."
echo ""
