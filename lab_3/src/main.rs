use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul};
use std::rc::Rc;
use std::time::{Duration, Instant};
use crate::delay_gen::DelayGen;
use crate::simulation_delay::SimulationDelay;

pub mod delay_gen {
    use rand::distributions::Distribution;
    use crate::TimeDur;
    use rand::thread_rng;
    use rand_distr::{Exp, Normal, Uniform};

    #[derive(Clone, Copy, Debug)]
    pub enum DelayGen {
        Normal(Normal<f64>),
        Uniform(Uniform<f64>),
        Exponential(Exp<f64>),
    }

    impl DelayGen {
        pub fn sample(&self) -> TimeDur {
            TimeDur(
                match self {
                    Self::Normal(dist) => dist.sample(&mut thread_rng()).round() as u64,
                    Self::Uniform(dist) => dist.sample(&mut thread_rng()).round() as u64,
                    Self::Exponential(dist) => dist.sample(&mut thread_rng()).round() as u64,
                }
            )
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct TimeDur(u64);

impl AddAssign for TimeDur {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
struct QueueSize(u64);
impl QueueSize {
    fn increment(&mut self) {
        self.0 += 1;
    }
    fn decrement(&mut self) {
        self.0 -= 1;
    }
}

impl Mul<TimeDur> for QueueSize {
    type Output = QueueTimeDur;

    fn mul(self, rhs: TimeDur) -> Self::Output {
        QueueTimeDur(self.0 * rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
struct QueueTimeDur(u64);

impl Add for QueueTimeDur {
    type Output = QueueTimeDur;

    fn add(self, rhs: Self) -> Self::Output {
        QueueTimeDur(self.0 + rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
struct MeanQueueSize(f64);

impl Div<TimeDur> for QueueTimeDur {
    type Output = MeanQueueSize;

    fn div(self, rhs: TimeDur) -> Self::Output {
        MeanQueueSize(self.0 as f64 / rhs.0 as f64)
    }
}

mod simulation_delay {
    use crate::{QueueSize, QueueTimeDur, TimeDur};
    use crate::delay_gen::DelayGen;

    #[derive(Debug, Clone, Copy)]
    pub struct SimulationDelay {
        work_dur: TimeDur,
        items_processed: u64,
        queue_time_dur: QueueTimeDur
    }

    impl SimulationDelay {
        pub fn new(delay_gen: DelayGen, queue_size: QueueSize) -> Self {
            let work_dur = delay_gen.sample();
            Self{work_dur, items_processed: 1, queue_time_dur: queue_size * work_dur}
        }

        pub fn combine(&mut self, simulation_delay: SimulationDelay) {
            self.work_dur += simulation_delay.work_dur;
            self.items_processed += simulation_delay.items_processed;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SimulationStats(SimulationDelay);

impl SimulationStats {
    fn combine(&mut self, simulation_delay: SimulationDelay) {
        self.0.combine(simulation_delay);
    }
}

pub trait Element : Debug {
    fn simulate(&mut self) -> SimulationDelay;
}

type ElementNode = Rc<RefCell<dyn Element>>;

fn simulate_model(root: &mut dyn Element, dur: Duration) {
    let stop_time_point = Instant::now() + dur;
    while Instant::now() < stop_time_point {
        let _ = root.simulate();
    }
}

#[derive(Debug)]
struct ElementBase {
    name: &'static str,
    queue_size: QueueSize,
    max_queue_size: QueueSize,
    stats: SimulationStats,
    delay_gen: DelayGen
}

mod simulation_probability {
    use std::ops::Deref;
    use crate::{Element, ElementBase, SimulationStats, SimulationDelay};
    use crate::simulation_probability::probability_elements::ProbabilityElements;

    pub mod probability {
        use std::time::Instant;

        #[derive(Copy, Clone, Debug)]
        pub struct Probability(f64);

        impl Probability {
            pub fn new(value: f64) -> Self {
                if value < 0.0 || value > 1.0 {
                    panic!("Probability can not be < 0.0 or > 1.0");
                }
                Self(value)
            }

            pub fn get_value(self) -> f64 {
                self.0
            }
        }
    }

    pub mod probability_elements {
        use std::cell::{RefCell};
        use std::rc::Rc;
        use rand::random;
        use crate::{Element, ElementNode};
        use super::probability::Probability;

        type InnerType = Vec<(ElementNode, Probability)>;

        #[derive(Default, Debug)]
        pub struct ProbabilityElements(InnerType);

        fn calc_prob_sum(probs: &InnerType) -> f64 {
            probs.iter().map(|e| (*e).1.get_value()).sum::<f64>()
        }

        impl ProbabilityElements {
            pub fn new(probs: InnerType) -> Self {
                assert!((calc_prob_sum(&probs) - 1.0).abs() < f64::EPSILON, "Sum of the probabilities must be 1");
                Self(probs)
            }

            pub fn sample(&self) -> Option<Rc<RefCell<dyn Element>>> {
                if self.0.is_empty() {
                    return None;
                }

                let rand_value = random::<f64>() * calc_prob_sum(&self.0);
                let mut current_sum = 0.0;

                let mut target_index = self.0.len() - 1;
                for (index, (_, prob)) in self.0.iter().enumerate() {
                    current_sum += prob.get_value();
                    if rand_value < current_sum {
                        target_index = index;
                        break;
                    }
                }
                Some(self.0.iter().nth(target_index)?.0.clone())
            }
        }
    }

    #[derive(Debug)]
    pub struct ElementProbabilityBase {
        base: ElementBase,
        children: ProbabilityElements,
    }

    impl Element for ElementProbabilityBase {
        fn simulate(&mut self) -> SimulationDelay {
            let mut delay = SimulationDelay::new(self.base.delay_gen, self.base.queue_size);
            if let Some(child) = self.children.sample() {
                delay.combine(child.borrow_mut().simulate());
            }
            self.base.stats.combine(delay);
            println!("{:?}", self);
            delay
        }
    }
}

mod simulation_priority {
    use crate::{Element, ElementBase, SimulationDelay};
    use crate::simulation_priority::priority_elements::PriorityElements;

    mod priority_elements {
        use std::slice::Iter;
        use crate::ElementNode;

        #[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
        struct Priority(u64);

        type InnerType = Vec<(ElementNode, Priority)>;

        #[derive(Default, Debug)]
        pub struct PriorityElements(InnerType);

        impl PriorityElements {
            pub fn new(mut priors: Vec<(ElementNode, Priority)>) -> Self {
                priors.sort_by(|a, b| (*a).1.cmp(&(*b).1));
                Self(priors)
            }

            pub fn iter(&self) -> PriorityElementsIterator {
                PriorityElementsIterator(self.0.iter())
            }
        }

        pub struct PriorityElementsIterator<'a>(Iter<'a, (ElementNode, Priority)>);

        impl<'a> Iterator for PriorityElementsIterator<'a> {
            type Item = ElementNode;

            fn next(&mut self) -> Option<Self::Item> {
                Some((*self.0.next()?).0.clone())
            }
        }
    }

    #[derive(Debug)]
    pub struct ElementPriorityBase {
        base: ElementBase,
        children: PriorityElements,
    }

    impl Element for ElementPriorityBase {
        fn simulate(&mut self) -> SimulationDelay {
            let mut delay = SimulationDelay::new(self.base.delay_gen, self.base.queue_size);
            for child in self.children.iter() {
                delay.combine(child.borrow_mut().simulate());
            }
            self.base.stats.combine(delay);
            println!("{:?}", self);
            delay
        }
    }
}


fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_general() {

    }
}


