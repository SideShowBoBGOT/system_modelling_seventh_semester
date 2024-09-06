mod simulation_probability {
    use std::ops::Deref;
    use crate::{Element, ElementBase, Payload, SimulationDelay, TimePoint};
    use crate::simulation_probability::probability_elements::ProbabilityElements;

    pub mod probability {
        #[derive(Copy, Clone, Debug, Default)]
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

    // #[derive(Debug)]
    // pub struct ElementProbabilityBase {
    //     base: ElementBase,
    //     children: ProbabilityElements,
    // }
    //
    // impl Element for ElementProbabilityBase {
    //     fn simulate(&mut self, payload: Payload, current_time: TimePoint) -> SimulationDelay {
    //         let mut delay = SimulationDelay::new(self.base.delay_gen, self.base.queue_size);
    //
    //
    //         if let Some(child) = self.children.sample() {
    //             delay.combine(child.borrow_mut().simulate(payload, current_time));
    //         }
    //         self.base.stats.combine(delay);
    //         println!("{:?}", self);
    //         delay
    //     }
    // }
}



mod simulation_priority {
    use crate::{Element, ElementBase, Payload, TimePoint};
    use crate::delay_gen::DelayGen;
    use crate::simulation_priority::payload_vec::PayloadQueue;
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
}
