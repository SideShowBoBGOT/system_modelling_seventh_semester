pub mod prob {
    #[derive(Copy, Clone, Debug)]
    pub struct Probability(f64);

    impl Probability {
        pub fn new(value: f64) -> Self {
            if value < 0.0 || value > 1.0 {
                panic!("Probability can not be < 0.0 or > 1.0");
            }
            Self(value)
        }

        pub fn get_value(&self) -> f64 {
            self.0
        }
    }
}

trait Element {

}

pub mod prob_el_map {
    use std::cell::{RefCell};
    use std::rc::Rc;
    use rand::random;
    use crate::{Element, ElementType};
    use crate::prob::Probability;

    type InnerType = Vec<(Rc<RefCell<dyn Element>>, Probability)>;

    #[derive(Default, Debug)]
    pub struct ProbabilityElements(InnerType);

    fn calc_prob_sum(probs: &InnerType) -> f64 {
        probs.iter().map(|e| (*e).1.get_value()).sum::<f64>()
    }

    impl ProbabilityElements {
        pub fn new(probs: Vec<(Rc<RefCell<ElementType>>, Probability)>) -> Self {
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

enum ElementType {
    ElementBase(ElementBase)
}

struct ElementBase {
    children: Vec<ElementType>
}



fn main() {
    println!("Hello, world!");
}


