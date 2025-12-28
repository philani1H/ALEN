//! Biologically-Inspired Emotion System Demo
//!
//! Demonstrates how ALEN processes emotions like a human brain:
//! - Limbic system (amygdala, hippocampus, hypothalamus)
//! - Neurotransmitter dynamics
//! - Prefrontal cortex regulation
//! - Emotional memory and learning

use alen::control::{
    EmotionSystem, EmotionalStimulus, StimulusType,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        ALEN Biologically-Inspired Emotion System            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut emotion_system = EmotionSystem::new();

    println!("Emotion system initialized with:");
    println!("  • Limbic system (amygdala, hippocampus, hypothalamus)");
    println!("  • Neurotransmitter dynamics (dopamine, serotonin, etc.)");
    println!("  • Prefrontal cortex (rational evaluation)");
    println!("  • Emotional memory\n");

    println!("═══════════════════════════════════════════════════════════════");
    println!("              SCENARIO 1: SUCCESS AND REWARD");
    println!("═══════════════════════════════════════════════════════════════\n");

    let success_stimulus = EmotionalStimulus {
        stimulus_type: StimulusType::Success,
        intensity: 0.9,
        valence: 0.8,
        context: "Solved complex problem correctly".to_string(),
    };

    println!("Stimulus: Task completed successfully");
    println!("  Type: Success");
    println!("  Intensity: 0.9");
    println!("  Valence: +0.8 (positive)\n");

    let response = emotion_system.process(success_stimulus);
    
    println!("Limbic Response:");
    println!("  Original emotion: {}", response.original_emotion.as_str());
    println!("  Regulation applied: {}", response.regulation_applied);
    println!("  Final emotion: {}", response.regulated_emotion.as_str());
    println!("  Rationale: {}\n", response.rationale);

    println!("═══════════════════════════════════════════════════════════════");
    println!("              SCENARIO 2: FAILURE AND STRESS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let failure_stimulus = EmotionalStimulus {
        stimulus_type: StimulusType::Failure,
        intensity: 0.8,
        valence: -0.7,
        context: "Failed to verify solution".to_string(),
    };

    println!("Stimulus: Task failed verification");
    println!("  Type: Failure");
    println!("  Intensity: 0.8");
    println!("  Valence: -0.7 (negative)\n");

    let response = emotion_system.process(failure_stimulus);
    
    println!("Limbic Response:");
    println!("  Original emotion: {}", response.original_emotion.as_str());
    println!("  Regulation applied: {}", response.regulation_applied);
    println!("  Final emotion: {}", response.regulated_emotion.as_str());
    println!("  Rationale: {}\n", response.rationale);

    println!("═══════════════════════════════════════════════════════════════");
    println!("              SCENARIO 3: THREAT DETECTION");
    println!("═══════════════════════════════════════════════════════════════\n");

    let threat_stimulus = EmotionalStimulus {
        stimulus_type: StimulusType::Threat,
        intensity: 0.95,
        valence: -0.9,
        context: "Detected potential system error".to_string(),
    };

    println!("Stimulus: Potential system error detected");
    println!("  Type: Threat");
    println!("  Intensity: 0.95");
    println!("  Valence: -0.9 (very negative)\n");

    let response = emotion_system.process(threat_stimulus);
    
    println!("Limbic Response:");
    println!("  Original emotion: {}", response.original_emotion.as_str());
    println!("  Regulation applied: {}", response.regulation_applied);
    println!("  Final emotion: {}", response.regulated_emotion.as_str());
    println!("  Rationale: {}\n", response.rationale);

    println!("═══════════════════════════════════════════════════════════════");
    println!("              SCENARIO 4: NOVEL DISCOVERY");
    println!("═══════════════════════════════════════════════════════════════\n");

    let novel_stimulus = EmotionalStimulus {
        stimulus_type: StimulusType::Novel,
        intensity: 0.7,
        valence: 0.6,
        context: "Discovered new pattern in data".to_string(),
    };

    println!("Stimulus: Discovered new pattern");
    println!("  Type: Novel");
    println!("  Intensity: 0.7");
    println!("  Valence: +0.6 (positive)\n");

    let response = emotion_system.process(novel_stimulus);
    
    println!("Limbic Response:");
    println!("  Original emotion: {}", response.original_emotion.as_str());
    println!("  Regulation applied: {}", response.regulation_applied);
    println!("  Final emotion: {}", response.regulated_emotion.as_str());
    println!("  Rationale: {}\n", response.rationale);

    println!("═══════════════════════════════════════════════════════════════");
    println!("              SCENARIO 5: SURPRISE");
    println!("═══════════════════════════════════════════════════════════════\n");

    let surprise_stimulus = EmotionalStimulus {
        stimulus_type: StimulusType::Surprise,
        intensity: 0.85,
        valence: 0.0,
        context: "Unexpected result from prediction".to_string(),
    };

    println!("Stimulus: Unexpected outcome");
    println!("  Type: Surprise");
    println!("  Intensity: 0.85");
    println!("  Valence: 0.0 (neutral)\n");

    let response = emotion_system.process(surprise_stimulus);
    
    println!("Limbic Response:");
    println!("  Original emotion: {}", response.original_emotion.as_str());
    println!("  Regulation applied: {}", response.regulation_applied);
    println!("  Final emotion: {}", response.regulated_emotion.as_str());
    println!("  Rationale: {}\n", response.rationale);

    println!("═══════════════════════════════════════════════════════════════");
    println!("              EMOTIONAL TREND ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let trend = emotion_system.emotional_trend();
    println!("Recent emotional history (most recent first):");
    for (i, emotion) in trend.iter().enumerate() {
        println!("  {}: {}", i + 1, emotion.as_str());
    }
    println!();

    println!("Current emotional state: {}\n", emotion_system.current_emotion().as_str());

    println!("═══════════════════════════════════════════════════════════════");
    println!("              CONTINUOUS PROCESSING SIMULATION");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Simulating continuous emotional processing...\n");

    let scenarios = vec![
        ("Success", StimulusType::Success, 0.8, 0.7),
        ("Reward", StimulusType::Reward, 0.9, 0.8),
        ("Failure", StimulusType::Failure, 0.6, -0.5),
        ("Success", StimulusType::Success, 0.7, 0.6),
        ("Novel", StimulusType::Novel, 0.8, 0.5),
        ("Familiar", StimulusType::Familiar, 0.3, 0.2),
        ("Threat", StimulusType::Threat, 0.9, -0.8),
        ("Success", StimulusType::Success, 0.8, 0.7),
    ];

    for (i, (name, stim_type, intensity, valence)) in scenarios.iter().enumerate() {
        let stimulus = EmotionalStimulus {
            stimulus_type: stim_type.clone(),
            intensity: *intensity,
            valence: *valence,
            context: format!("Event {}", i + 1),
        };

        let response = emotion_system.process(stimulus);
        
        println!("Event {}: {} → {} (regulated: {})",
            i + 1,
            name,
            response.regulated_emotion.as_str(),
            if response.regulation_applied { "yes" } else { "no" }
        );
    }

    println!("\nFinal emotional state: {}", emotion_system.current_emotion().as_str());

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    KEY INSIGHTS                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("How ALEN's Emotion System Works:");
    println!();
    println!("1. SENSORY INPUT → LIMBIC SYSTEM");
    println!("   • Stimulus detected (success, failure, threat, etc.)");
    println!("   • Amygdala evaluates emotional salience");
    println!("   • Hippocampus checks emotional memory");
    println!();
    println!("2. NEUROTRANSMITTER RELEASE");
    println!("   • Hypothalamus triggers chemical responses:");
    println!("     - Success → ↑ dopamine, ↑ serotonin");
    println!("     - Threat → ↑ norepinephrine, ↑ cortisol");
    println!("     - Novel → ↑ dopamine, ↑ glutamate");
    println!();
    println!("3. EMOTIONAL STATE COMPUTATION");
    println!("   • Valence = f(dopamine, serotonin, cortisol)");
    println!("   • Arousal = f(norepinephrine, glutamate, GABA)");
    println!("   • Emotion emerges from network activation");
    println!();
    println!("4. PREFRONTAL CORTEX EVALUATION");
    println!("   • Rational assessment of emotional response");
    println!("   • Cognitive reappraisal if needed");
    println!("   • \"Thinking yourself out of\" strong emotions");
    println!();
    println!("5. BEHAVIORAL OUTPUT");
    println!("   • Regulated emotion influences decisions");
    println!("   • Stored in emotional memory");
    println!("   • Neurotransmitters decay over time (homeostasis)");
    println!();
    println!("This is NOT hardcoded responses - emotions EMERGE from");
    println!("network dynamics, just like in biological brains!");
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              EMOTION SYSTEM DEMONSTRATION COMPLETE           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
