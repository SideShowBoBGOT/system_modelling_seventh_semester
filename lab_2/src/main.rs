use crate::delay_gen::DelayGen;
use rand::Rng;
use rand_distr::Distribution;
use std::cell::RefMut;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

type TimeUnit = u64;

pub mod delay_gen {
    use crate::TimeUnit;
    use rand::distributions::Distribution;
    use rand::thread_rng;
    use rand_distr::{Exp, Normal, Uniform};

    #[derive(Clone, Copy, Debug)]
    pub enum DelayGen {
        Normal(Normal<f64>),
        Uniform(Uniform<f64>),
        Exponential(Exp<f64>),
    }

    impl DelayGen {
        pub fn sample(&self) -> TimeUnit {
            match self {
                Self::Normal(dist) => dist.sample(&mut thread_rng()).round() as TimeUnit,
                Self::Uniform(dist) => dist.sample(&mut thread_rng()).round() as TimeUnit,
                Self::Exponential(dist) => dist.sample(&mut thread_rng()).round() as TimeUnit,
            }
        }
    }
}

pub trait Element : Debug {
    fn next_element(&mut self) -> Option<RefMut<dyn Element>>;
    fn in_act(&mut self);
    fn out_act(&mut self);
    fn set_current_t(&mut self, current_t: TimeUnit);
    fn get_next_t(&self) -> TimeUnit;
    fn calc_statistic(&mut self, delta: TimeUnit);
}

pub mod prob_el_map {
    use crate::Element;
    use rand::random;
    use std::cell::{RefCell, RefMut};
    use std::collections::HashMap;
    use std::fmt::{Debug, Formatter};
    use std::rc::Rc;

    #[derive(Default)]
    pub struct ProbabilityElementsMap(HashMap<Rc<RefCell<dyn Element>>, f64>);

    impl Debug for ProbabilityElementsMap {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "")
        }
    }

    impl ProbabilityElementsMap {
        pub fn new(next_elements_map: HashMap<Rc<RefCell<dyn Element>>, f64>) -> Self {
            const EPSILON: f64 = 0.001;
            assert!(next_elements_map.values().all(|v| *v >= 0.0 && *v <= 1.0), "Values must be between 0.0 and 1.0");
            assert!((next_elements_map.values().sum::<f64>() - 1.0).abs() < EPSILON);
            Self(next_elements_map)
        }

        pub fn sample(&mut self) -> Option<RefMut<dyn Element>> {
            if self.0.is_empty() {
                return None;
            }

            let total_prob: f64 = self.0.values().sum();
            let rand_value = random::<f64>() * total_prob;
            let mut current_sum = 0.0;

            let mut target_index = self.0.len() - 1;
            for (index, (_el, prob)) in self.0.iter().enumerate() {
                current_sum += *prob;
                if rand_value < current_sum {
                    target_index = index;
                    break;
                }
            }
            Some(self.0.iter_mut().nth(target_index)?.0.borrow_mut())
        }
    }
}

#[derive(Debug)]
struct ElementBase {
    name: &'static str,
    next_t: TimeUnit,
    current_t: TimeUnit,
    delay_gen: DelayGen,
    quantity: usize,
}

impl ElementBase {
    fn new(name: &'static str, delay_gen: DelayGen) -> Self {
        Self{name, next_t: 0, current_t: 0, delay_gen, quantity: 0}
    }
}

pub mod element_create {
    use crate::delay_gen::DelayGen;
    use crate::prob_el_map::ProbabilityElementsMap;
    use crate::{Element, ElementBase, TimeUnit};
    use std::cell::RefMut;
    use std::fmt::Debug;

    #[derive(Debug)]
    pub struct ElementCreate {
        base: ElementBase,
        next_elements: ProbabilityElementsMap
    }

    impl ElementCreate {
        pub fn new(
            name: &'static str, delay_gen: DelayGen, next_elements: ProbabilityElementsMap
        ) -> Self {
            Self{base: ElementBase::new(name, delay_gen), next_elements}
        }
    }

    impl Element for ElementCreate {
        fn next_element(&mut self) -> Option<RefMut<dyn Element>> {
            self.next_elements.sample()
        }
        fn in_act(&mut self) {}
        fn out_act(&mut self) {
            self.base.quantity += 1;
            self.base.next_t = self.base.current_t + self.base.delay_gen.sample();
        }
        fn set_current_t(&mut self, current_t: TimeUnit) {
            self.base.current_t = current_t;
        }
        fn get_next_t(&self) -> TimeUnit {
            self.base.next_t
        }
        fn calc_statistic(&mut self, _: TimeUnit) {}
    }
}

pub mod device {
    use crate::delay_gen::DelayGen;
    use crate::TimeUnit;

    #[derive(Debug)]
    struct BaseData {
        name: &'static str,
        next_t: TimeUnit,
        quantity: usize,
        delay_gen: DelayGen,
    }

    #[derive(Debug)]
    pub struct FreeState(BaseData);
    impl FreeState {
        pub fn in_act(&self) -> BusyState {
            BusyState(
                BaseData{
                    name: self.0.name,
                    next_t: self.0.next_t + self.0.delay_gen.sample(),
                    quantity: self.0.quantity,
                    delay_gen: self.0.delay_gen
                }
            )
        }
        pub fn get_next_t(&self) -> TimeUnit {
            self.0.next_t
        }
    }
    #[derive(Debug)]
    pub struct BusyState(BaseData);
    impl BusyState {
        pub fn out_act(&self) -> FreeState {
            FreeState(
                BaseData {
                    name: self.0.name,
                    next_t: TimeUnit::MAX,
                    quantity: self.0.quantity + 1,
                    delay_gen: self.0.delay_gen
                }
            )
        }
        pub fn get_next_t(&self) -> TimeUnit {
            self.0.next_t
        }
    }

    #[derive(Debug)]
    pub enum Device {
        Free(FreeState),
        Busy(BusyState)
    }

    impl Device {
        pub fn new(name: &'static str, delay_gen: DelayGen) -> Self {
            Device::Free(FreeState(BaseData{name, next_t: 0, quantity: 0, delay_gen}))
        }

        pub fn get_next_t(&self) -> TimeUnit {
            match self {
                Self::Free(s) => s.get_next_t(),
                Self::Busy(s) => s.get_next_t(),
            }
        }
    }
}

mod element_process {
    use crate::delay_gen::DelayGen;
    use crate::device::Device;
    use crate::prob_el_map::ProbabilityElementsMap;
    use crate::{Element, ElementBase, TimeUnit};
    use std::cell::RefMut;

    #[derive(Debug)]
    pub struct ElementProcess {
        base: ElementBase,
        next_elements: ProbabilityElementsMap,
        queue: usize,
        max_queue: usize,
        count_rejected: usize,
        mean_queue: f64,
        devices: Vec<Device>
    }

    impl ElementProcess {
        pub fn new(name: &'static str, delay_gen: DelayGen, next_elements: ProbabilityElementsMap, max_queue: usize, devices: Vec<Device>) -> Self {
            Self{base: ElementBase::new(name, delay_gen), next_elements, queue: 0, max_queue, count_rejected: 0, mean_queue: 0.0, devices}
        }
    }

    fn out_act_busy_devices(devices: &mut [Device], quantity: &mut usize) {
        for device in devices {
            match device {
                Device::Free(_) => (),
                Device::Busy(busy_state) => {
                    *device = Device::Free(busy_state.out_act());
                    *quantity += 1;
                },
            }
        }
    }

    fn get_min_next_t(devices: &[Device]) -> TimeUnit {
        devices.iter().map(|d| d.get_next_t()).min().unwrap()
    }

    fn find_free_device_index(devices: &[Device]) -> Option<usize> {
        devices.iter().enumerate().find(|d| {
            match d.1 {
                Device::Free(_) => true,
                Device::Busy(_) => false
            }
        }).map(|d| d.0)
    }

    impl Element for ElementProcess {
        fn next_element(&mut self) -> Option<RefMut<dyn Element>> {
            self.next_elements.sample()
        }
        fn in_act(&mut self) {
            if let Some(free_device_index) = find_free_device_index(&self.devices) {
                let free_device = &mut self.devices[free_device_index];
                match free_device {
                    Device::Free(s) => {
                        let busy_state = s.in_act();
                        self.base.next_t = busy_state.get_next_t();
                        *free_device = Device::Busy(busy_state);
                    },
                    _ => unreachable!()
                }
            } else if(self.queue < self.max_queue) {
                self.queue += 1;
            } else {
                self.count_rejected += 1;
            }
        }

        fn out_act(&mut self) {
            out_act_busy_devices(&mut self.devices, &mut self.base.quantity);
            while let Some(free_device_index) = find_free_device_index(&self.devices) {
                if self.queue == 0 {
                    break;
                }
                let free_device = &mut self.devices[free_device_index];
                match free_device {
                    Device::Free(s) => {
                        *free_device = Device::Busy(s.in_act());
                        self.queue -= 1;
                    },
                    Device::Busy(_) => unreachable!()
                }
            }

            self.base.next_t = get_min_next_t(&self.devices);
        }

        fn set_current_t(&mut self, current_t: TimeUnit) {
            self.base.current_t = current_t;
        }
        fn get_next_t(&self) -> TimeUnit {
            self.base.next_t
        }
        fn calc_statistic(&mut self, delta: TimeUnit) {
            self.mean_queue += (self.queue as u64 * delta) as f64;
        }
    }
}


fn simulate_model(elements: &mut [&mut dyn Element], max_time: TimeUnit) {
    let mut current_t: TimeUnit = 0;
    while current_t < max_time {
        let (element_index, next_t) = elements.iter()
            .map(|d| d.get_next_t())
            .enumerate()
            .min_by(|d1, d2| (*d1).1.cmp(&(*d2).1))
            .expect("elements can not be empty");
        println!("Current event {}, {}", element_index, next_t);
        for el in &mut *elements {
            el.calc_statistic(next_t - current_t)
        }
        current_t = next_t;
        for el in &mut *elements {
            el.set_current_t(current_t)
        }
        elements.iter_mut().nth(element_index).unwrap().out_act();
        let _ = elements.iter_mut()
            .filter(|e| e.get_next_t() == current_t)
            .map(|e| e.out_act());
        for el in &mut *elements {
            println!("{:?}", el);
        }
    }
}


fn main() {}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use log::Level::Debug;
    use crate::device::Device;
    use crate::element_process::ElementProcess;
    use crate::prob_el_map::ProbabilityElementsMap;
    use super::*;

    #[test]
    fn test_general() {
        let delay_gen = DelayGen::Uniform(rand_distr::Uniform::<f64>::new(0., 10.));
        let element_process_3 = Rc::new(RefCell::new(ElementProcess::new(
            "Process 3", delay_gen, Default::default(), 4,
            vec![Device::new("Device 1", delay_gen)]
        )));
        let element_process_2 = Rc::new(RefCell::new(ElementProcess::new(
            "Process 2", delay_gen, Default::default(), 3,
            vec![
                Device::new("Device 1", delay_gen),
                Device::new("Device 2", delay_gen),
                Device::new("Device 3", delay_gen),
            ]
        )));
        let element_process_1 = Rc::new(RefCell::new(ElementProcess::new(
            "Process 1", delay_gen,
            ProbabilityElementsMap::new(
                HashMap::from([
                    (element_process_3.clone(), 0.25),
                    (element_process_2.clone(), 0.75),
                ])
            ),
            3,
            vec![
                Device::new("Device 1", delay_gen),
                Device::new("Device 2", delay_gen),
                Device::new("Device 3", delay_gen),
            ]
        )));

        let element_create = element_create::ElementCreate::new(
            "Create", delay_gen,
            ProbabilityElementsMap::new(
                HashMap::from([
                    (element_process_1.clone(), 1.0),
                ])
            )
        );


    }

    #[test]
    fn test_add_negative() {
    }
}

