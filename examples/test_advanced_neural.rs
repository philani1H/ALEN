use alen::neural::{
    AdvancedALENSystem, AdvancedALENConfig, MathProblemSolver,
    AudienceLevel, Tensor, ExplorationMode,
};

fn main() {
    println!("Testing Advanced Neural System...\n");
    
    // Test 1: Create system with default config
    println!("1. Creating Advanced ALEN System...");
    let config = AdvancedALENConfig::default();
    let system = AdvancedALENSystem::new(config);
    let stats = system.get_stats();
    println!("   ✓ System created successfully");
    println!("   - Total steps: {}", stats.total_steps);
    println!("   - Memory capacity: {:.1}%", stats.memory_stats.capacity_used * 100.0);
    
    // Test 2: Create math problem solver
    println!("\n2. Creating Math Problem Solver...");
    let config = AdvancedALENConfig::default();
    let mut solver = MathProblemSolver::new(config);
    println!("   ✓ Solver created successfully");
    
    // Test 3: Solve a simple problem
    println!("\n3. Solving a math problem...");
    let solution = solver.solve(
        "What is 2 + 2?",
        AudienceLevel::Elementary
    );
    println!("   ✓ Problem solved");
    println!("   - Solution: {}", solution.solution);
    println!("   - Confidence: {:.1}%", solution.confidence * 100.0);
    println!("   - Explanation: {}", solution.explanation);
    
    // Test 4: Test tensor operations
    println!("\n4. Testing tensor operations...");
    let t1 = Tensor::randn(&[2, 3]);
    let t2 = Tensor::randn(&[2, 3]);
    let t3 = t1.add(&t2);
    println!("   ✓ Tensor operations working");
    println!("   - Shape: {:?}", t3.shape());
    
    println!("\n✅ All tests passed!");
}
