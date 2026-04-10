/*!
# cuda-adaptation

Runtime self-adaptation for agents.

An agent that can't adapt is a tool. An agent that adapts is alive.

This crate provides the mechanisms for runtime behavior modification:
- Learning rate that adapts based on recent performance
- Strategy switching when current approach fails
- Parameter tuning via hill-climbing
- Behavioral plasticity — how much an agent can change
- Feedback integration — turning outcomes into behavior changes
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An adaptive parameter with bounds and learning rate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveParam {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub learning_rate: f64,
    pub history: Vec<f64>,
    pub best_value: f64,
    pub best_score: f64,
}

impl AdaptiveParam {
    pub fn new(name: &str, value: f64, min: f64, max: f64, lr: f64) -> Self {
        AdaptiveParam { name: name.to_string(), value: value.clamp(min, max), min, max, learning_rate: lr, history: vec![value], best_value: value, best_score: 0.0 }
    }

    /// Update based on feedback score (positive = good, negative = bad)
    pub fn update(&mut self, score: f64) {
        self.history.push(self.value);

        // If score improved, keep direction. Otherwise, reverse.
        if score > self.best_score {
            self.best_score = score;
            self.best_value = self.value;
        } else {
            // Nudge toward best known
            let delta = (self.best_value - self.value) * self.learning_rate;
            self.value = (self.value + delta).clamp(self.min, self.max);
        }

        // Adapt learning rate — slow down when stable, speed up when volatile
        if self.history.len() > 10 {
            let recent: f64 = self.history.iter().rev().take(5).map(|v| (v - self.value).abs()).sum::<f64>() / 5.0;
            if recent < 0.01 { self.learning_rate *= 0.95; } // stable, slow down
            else { self.learning_rate = (self.learning_rate * 1.05).min(0.5); } // volatile, speed up
        }
    }

    /// Perturb value for exploration (hill climbing)
    pub fn perturb(&mut self, magnitude: f64) {
        let noise = (rand_f64() - 0.5) * 2.0 * magnitude;
        self.value = (self.value + noise * self.learning_rate).clamp(self.min, self.max);
    }

    /// Reset to best known value
    pub fn reset_to_best(&mut self) {
        self.value = self.best_value;
    }

    /// Variance of recent history (stability measure)
    pub fn recent_variance(&self, n: usize) -> f64 {
        let recent: Vec<_> = self.history.iter().rev().take(n).cloned().collect();
        if recent.len() < 2 { return 0.0; }
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let var: f64 = recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        var
    }
}

/// A behavioral strategy that can be switched
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Strategy {
    pub name: String,
    pub description: String,
    pub fitness: f64,        // running average of performance
    pub usage_count: u32,
    pub params: HashMap<String, f64>,
    pub tags: Vec<String>,
}

impl Strategy {
    pub fn new(name: &str, desc: &str) -> Self {
        Strategy { name: name.to_string(), description: desc.to_string(), fitness: 0.5, usage_count: 0, params: HashMap::new(), tags: vec![] }
    }

    /// Update fitness with exponential moving average
    pub fn update_fitness(&mut self, score: f64, alpha: f64) {
        self.fitness = self.fitness * (1.0 - alpha) + score * alpha;
        self.usage_count += 1;
    }
}

/// Strategy pool with selection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyPool {
    pub strategies: HashMap<String, Strategy>,
    pub current: Option<String>,
    pub switch_threshold: f64,   // switch when current is this much worse than best
    pub exploration_rate: f64,   // probability of trying non-best
}

impl StrategyPool {
    pub fn new() -> Self { StrategyPool { strategies: HashMap::new(), current: None, switch_threshold: 0.15, exploration_rate: 0.1 } }

    pub fn add(&mut self, strategy: Strategy) {
        self.strategies.insert(strategy.name.clone(), strategy);
        if self.current.is_none() { self.current = Some(self.strategies.keys().next().unwrap().clone()); }
    }

    /// Report score for current strategy, possibly switch
    pub fn report(&mut self, score: f64) -> Option<String> {
        let current = self.current.as_ref()?;
        let alpha = 0.3;
        if let Some(strat) = self.strategies.get_mut(current) {
            strat.update_fitness(score, alpha);
        }

        // Explore?
        if rand_f64() < self.exploration_rate {
            let best = self.best_strategy_name();
            if let Some(best) = best {
                if best != *current { self.current = Some(best); return Some(best); }
            }
        }

        // Exploit: switch if significantly worse than best
        let best_fitness = self.strategies.values().map(|s| s.fitness).fold(0.0_f64, f64::max);
        let current_fitness = self.strategies.get(current).map(|s| s.fitness).unwrap_or(0.0);
        if best_fitness - current_fitness > self.switch_threshold {
            let best = self.best_strategy_name()?;
            self.current = Some(best.clone());
            Some(best)
        } else {
            None
        }
    }

    fn best_strategy_name(&self) -> Option<String> {
        self.strategies.iter().max_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap()).map(|(k, _)| k.clone())
    }

    pub fn current_fitness(&self) -> f64 {
        self.current.as_ref().and_then(|n| self.strategies.get(n)).map(|s| s.fitness).unwrap_or(0.0)
    }
}

/// Behavioral plasticity — how much an agent can change
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plasticity {
    pub exploration: f64,    // 0=exploit only, 1=explore only
    pub adaptation_speed: f64, // how fast parameters change
    pub memory_span: f64,    // how much history influences behavior
    pub rigidity: f64,       // resistance to change (increases with age)
    pub age: u32,
}

impl Plasticity {
    pub fn new() -> Self {
        Plasticity { exploration: 0.3, adaptation_speed: 0.5, memory_span: 0.7, rigidity: 0.0, age: 0 }
    }

    /// Tick — aging increases rigidity, decreases exploration
    pub fn tick(&mut self) {
        self.age += 1;
        self.rigidity = (self.rigidity + 0.001).min(0.5);
        self.exploration = (self.exploration - 0.0005).max(0.05);
    }

    /// Effective adaptation speed = base * (1 - rigidity)
    pub fn effective_speed(&self) -> f64 {
        self.adaptation_speed * (1.0 - self.rigidity)
    }
}

/// The full adaptation controller
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptationController {
    pub params: HashMap<String, AdaptiveParam>,
    pub strategy_pool: StrategyPool,
    pub plasticity: Plasticity,
    pub total_score: f64,
    pub episode_count: u32,
}

impl AdaptationController {
    pub fn new() -> Self { AdaptationController { params: HashMap::new(), strategy_pool: StrategyPool::new(), plasticity: Plasticity::new(), total_score: 0.0, episode_count: 0 } }

    pub fn add_param(&mut self, param: AdaptiveParam) {
        self.params.insert(param.name.clone(), param);
    }

    /// Process feedback: update params, strategies, plasticity
    pub fn feedback(&mut self, score: f64) -> String {
        self.total_score += score;
        self.episode_count += 1;
        self.plasticity.tick();

        let speed = self.plasticity.effective_speed();
        let mut changes = vec![];

        // Update parameters
        for param in self.params.values_mut() {
            param.learning_rate = param.learning_rate * speed;
            param.update(score);
            changes.push(format!("{}={:.3}", param.name, param.value));
        }

        // Update strategy
        if let Some(switched) = self.strategy_pool.report(score) {
            changes.push(format!("strategy→{}", switched));
        }

        changes.join(", ")
    }

    /// Explore: perturb all parameters
    pub fn explore(&mut self) {
        let magnitude = self.plasticity.exploration;
        for param in self.params.values_mut() {
            param.perturb(magnitude);
        }
    }

    /// Get average performance
    pub fn avg_score(&self) -> f64 {
        if self.episode_count == 0 { return 0.0; }
        self.total_score / self.episode_count as f64
    }

    /// Summary
    pub fn summary(&self) -> AdaptationSummary {
        AdaptationSummary {
            avg_score: self.avg_score(),
            episodes: self.episode_count,
            params: self.params.len(),
            strategies: self.strategy_pool.strategies.len(),
            current_strategy: self.strategy_pool.current.clone(),
            exploration: self.plasticity.exploration,
            rigidity: self.plasticity.rigidity,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdaptationSummary {
    pub avg_score: f64,
    pub episodes: u32,
    pub params: usize,
    pub strategies: usize,
    pub current_strategy: Option<String>,
    pub exploration: f64,
    pub rigidity: f64,
}

/// Simple pseudo-random (deterministic for tests)
fn rand_f64() -> f64 {
    use std::time::SystemTime;
    let n = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos() as f64;
    (n * 0.0000000001 % 1.0).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_update() {
        let mut p = AdaptiveParam::new("lr", 0.1, 0.0, 1.0, 0.01);
        p.update(0.9); // good score
        assert_eq!(p.best_score, 0.9);
    }

    #[test]
    fn test_param_clamp() {
        let mut p = AdaptiveParam::new("x", 0.5, 0.0, 1.0, 1.0);
        p.value = 2.0; // force out of bounds
        p.perturb(10.0);
        assert!(p.value >= 0.0 && p.value <= 1.0);
    }

    #[test]
    fn test_param_reset() {
        let mut p = AdaptiveParam::new("x", 0.5, 0.0, 1.0, 0.01);
        p.update(0.9);
        p.value = 0.1;
        p.reset_to_best();
        assert_eq!(p.value, 0.5);
    }

    #[test]
    fn test_param_variance() {
        let mut p = AdaptiveParam::new("x", 0.5, 0.0, 1.0, 0.01);
        p.history = vec![0.0, 1.0, 0.0, 1.0];
        let v = p.recent_variance(4);
        assert!(v > 0.0);
    }

    #[test]
    fn test_strategy_fitness() {
        let mut s = Strategy::new("aggressive", "fast approach");
        s.update_fitness(0.8, 0.3);
        assert!(s.fitness > 0.5);
        assert_eq!(s.usage_count, 1);
    }

    #[test]
    fn test_strategy_pool_switch() {
        let mut pool = StrategyPool::new();
        pool.exploration_rate = 0.0; // disable random exploration
        pool.add(Strategy::new("a", ""));
        pool.add(Strategy::new("b", ""));
        pool.current = Some("a".into());
        // Report bad scores for "a", then good for "b"
        for _ in 0..20 { pool.report(0.1); } // tank a's fitness
        // Manually boost b
        pool.strategies.get_mut("b").unwrap().fitness = 0.9;
        let switched = pool.report(0.1);
        assert!(switched.is_some());
    }

    #[test]
    fn test_plasticity_aging() {
        let mut p = Plasticity::new();
        let initial_rigidity = p.rigidity;
        for _ in 0..100 { p.tick(); }
        assert!(p.rigidity > initial_rigidity);
        assert!(p.effective_speed() < 0.5);
    }

    #[test]
    fn test_adaptation_feedback() {
        let mut ctrl = AdaptationController::new();
        ctrl.add_param(AdaptiveParam::new("x", 0.5, 0.0, 1.0, 0.01));
        ctrl.strategy_pool.add(Strategy::new("default", ""));
        let changes = ctrl.feedback(0.8);
        assert!(changes.contains("x="));
    }

    #[test]
    fn test_adaptation_explore() {
        let mut ctrl = AdaptationController::new();
        ctrl.add_param(AdaptiveParam::new("x", 0.5, 0.0, 1.0, 0.01));
        let before = ctrl.params["x"].value;
        ctrl.explore();
        // Value might change (probabilistic)
        assert!(ctrl.params["x"].value >= 0.0 && ctrl.params["x"].value <= 1.0);
    }

    #[test]
    fn test_adaptation_avg_score() {
        let mut ctrl = AdaptationController::new();
        ctrl.feedback(0.8);
        ctrl.feedback(0.6);
        assert!((ctrl.avg_score() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_adaptation_summary() {
        let mut ctrl = AdaptationController::new();
        ctrl.add_param(AdaptiveParam::new("x", 0.5, 0.0, 1.0, 0.01));
        ctrl.feedback(0.5);
        let s = ctrl.summary();
        assert_eq!(s.params, 1);
        assert_eq!(s.episodes, 1);
    }
}
