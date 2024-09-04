use crate::delay_gen::DelayGen;
use crate::device::Device;
use crate::element_process::ElementProcess;
use crate::prob_el_map::ProbabilityElementsMap;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

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

pub mod prob_el_map {
    use crate::Element;
    use rand::random;
    use std::cell::{RefCell, RefMut};
    use std::fmt::{Debug, Formatter};
    use std::rc::Rc;

    #[derive(Default)]
    pub struct ProbabilityElementsMap(Vec<(Rc<RefCell<dyn Element>>, f64)>);

    impl Debug for ProbabilityElementsMap {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "")
        }
    }

    impl ProbabilityElementsMap {
        pub fn new(next_elements_map: Vec<(Rc<RefCell<dyn Element>>, f64)>) -> Self {
            const EPSILON: f64 = 0.001;
            assert!(
                next_elements_map.iter()
                    .map(|e| (*e).1).all(|v| v >= 0.0 && v <= 1.0),
                "Values must be between 0.0 and 1.0"
            );
            assert!((next_elements_map.iter().map(|e| (*e).1).sum::<f64>() - 1.0).abs() < EPSILON);
            Self(next_elements_map)
        }

        pub fn sample(&self) -> Option<RefMut<dyn Element>> {
            if self.0.is_empty() {
                return None;
            }

            let total_prob: f64 = self.0.iter().map(|e| (*e).1).sum();
            let rand_value = random::<f64>() * total_prob;
            let mut current_sum = 0.0;

            let mut target_index = self.0.len() - 1;
            for (index, (_, prob)) in self.0.iter().enumerate() {
                current_sum += *prob;
                if rand_value < current_sum {
                    target_index = index;
                    break;
                }
            }
            Some(self.0.iter().nth(target_index)?.0.borrow_mut())
        }
    }
}

pub trait Element : Debug {
    fn in_act(&mut self);
    fn out_act(&mut self);
    fn set_current_t(&mut self, current_t: TimeUnit);
    fn get_next_t(&self) -> TimeUnit;
    fn update_statistic(&mut self, next_t: TimeUnit, current_t: TimeUnit);
    fn update_next_t(&mut self);
    fn print_stats(&self);
}

pub struct ElementBase {
    name: &'static str,
    next_t: TimeUnit,
    current_t: TimeUnit,
    work_time: TimeUnit,
    quantity: usize,
    quantity_processed: usize,
    is_working: bool,
    delay_gen: DelayGen,
    next_elements: ProbabilityElementsMap,
}

impl Debug for ElementBase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f, "name: {:?}, next_t: {:?}, current_t: {:?}, quantity: {:?}, quantity_processed: {:?}, work_time: {:?}",
            self.name, self.next_t, self.current_t, self.quantity, self.quantity_processed, self.work_time
        )
    }
}

impl ElementBase {
    pub fn new(name: &'static str, delay_gen: DelayGen, next_elements: ProbabilityElementsMap) -> Self {
        Self{
            name, next_t: 0, current_t: 0, quantity: 0, work_time: 0,
            quantity_processed: 0, is_working: false,
            delay_gen, next_elements
        }
    }
}

impl ElementBase {
    fn in_act(&mut self) {
        self.quantity += 1;
        self.is_working = true;
    }

    fn out_act(&mut self) {
        self.quantity_processed += 1;
        self.is_working = false;
        if let Some(mut next_element) = self.next_elements.sample() {
            next_element.in_act();
        }
    }

    fn set_current_t(&mut self, current_t: TimeUnit) {
        self.current_t = current_t;
    }
    fn get_next_t(&self) -> TimeUnit {
        self.next_t
    }

    fn update_statistic(&mut self, next_t: TimeUnit, current_t: TimeUnit) {
        self.work_time += if self.is_working { next_t - current_t } else { 0 };
    }

    fn print_stats(&self) {
        println!("name = {}", self.name);
        println!("\tquantity_processed = {}", self.quantity_processed);
    }
}

#[derive(Debug)]
struct ElementCreate(ElementBase);

impl Element for ElementCreate {
    fn in_act(&mut self) {
        self.0.in_act()
    }

    fn out_act(&mut self) {
        self.0.out_act();
        self.update_next_t();
    }

    fn set_current_t(&mut self, current_t: TimeUnit) {
        self.0.set_current_t(current_t)
    }

    fn get_next_t(&self) -> TimeUnit {
        self.0.get_next_t()
    }

    fn update_statistic(&mut self, next_t: TimeUnit, current_t: TimeUnit) {
        self.0.update_statistic(next_t, current_t)
    }

    fn update_next_t(&mut self) {
        self.0.next_t = self.0.current_t + self.0.delay_gen.sample();
    }

    fn print_stats(&self) {
        self.0.print_stats()
    }
}

pub mod device {
    use crate::delay_gen::DelayGen;
    use crate::TimeUnit;

    #[derive(Debug)]
    pub struct Device {
        name: &'static str,
        next_t: TimeUnit,
        delay_gen: DelayGen,
        is_working: bool
    }

    impl Device {
        pub fn new(name: &'static str, delay_gen: DelayGen) -> Self {
            Self{name, next_t: 0, delay_gen, is_working: false}
        }

        pub fn in_act(&mut self, current_t: TimeUnit) {
            self.next_t = current_t + self.delay_gen.sample();
            self.is_working = true;
        }

        pub fn out_act(&mut self) {
            self.next_t = TimeUnit::MAX;
            self.is_working = false;
        }
        
        pub fn get_next_t(&self) -> TimeUnit {
            self.next_t
        }

        pub fn get_is_working(&self) -> bool {
            self.is_working
        }
    }
}

mod element_process {
    use crate::device::Device;
    use crate::{Element, ElementBase, TimeUnit};
    use std::fmt::{Debug, Formatter};

    pub struct ElementProcess {
        base: ElementBase,
        queue: usize,
        max_queue: usize,
        count_rejected: usize,
        mean_queue: f64,
        devices: Vec<Device>
    }

    impl Debug for ElementProcess {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(
                f, "{:?}, queue: {:?}, max_queue: {:?}, count_rejected: {:?}, mean_queue: {:?}",
                self.base, self.queue, self.max_queue, self.count_rejected, self.mean_queue
            )
        }
    }

    impl ElementProcess {
        pub fn new(mut base: ElementBase, max_queue: usize, devices: Vec<Device>) -> Self {
            base.next_t = TimeUnit::MAX;
            Self{base, queue: 0, max_queue, count_rejected: 0, mean_queue: 0.0, devices}
        }
    }

    fn find_free_device(devices: &mut [Device]) -> Option<&mut Device> {
        devices.iter_mut().find(|d| !(*d).get_is_working())
    }

    impl Element for ElementProcess {
        fn in_act(&mut self) {
            self.base.in_act();
            if let Some(free_device) = find_free_device(&mut self.devices) {
                free_device.in_act(self.base.current_t);
            } else if self.queue < self.max_queue {
                self.queue += 1;
            } else {
                self.count_rejected += 1;
            }
            self.update_next_t();
        }
        fn out_act(&mut self) {
            for device in &mut self.devices {
                device.out_act();
                self.base.out_act();
                if self.queue > 0 {
                    self.queue -= 1;
                    device.in_act(self.base.current_t);
                }
            }
            self.update_next_t();
        }

        fn set_current_t(&mut self, current_t: TimeUnit) {
            self.base.set_current_t(current_t);
        }

        fn get_next_t(&self) -> TimeUnit {
            self.base.get_next_t()
        }
        fn update_statistic(&mut self, next_t: TimeUnit, current_t: TimeUnit) {
            self.base.update_statistic(next_t, current_t);
            self.mean_queue += (self.queue as u64 * (next_t - current_t)) as f64;
        }
        fn update_next_t(&mut self) {
            self.base.next_t = self.devices.iter().map(|d| d.get_next_t()).min().unwrap()
        }
        fn print_stats(&self) {
            self.base.print_stats();
            println!("\twork_time = {}", self.base.work_time);
            println!("\tmean_queue = {}", self.mean_queue);
            println!("\tcount_rejected = {}", self.count_rejected);
            println!("\tqueue = {}", self.queue);
            println!("\tmax_queue = {}", self.max_queue);
            println!("\tfailure_probability = {}", self.count_rejected as f64 / self.base.quantity as f64);
        }
    }
}


fn simulate_model(mut elements: Vec<Rc<RefCell<dyn Element>>>, max_time: TimeUnit) {
    let mut current_t: TimeUnit = 0;
    while current_t < max_time {
        let next_t = elements.iter()
            .map(|d| d.borrow().get_next_t()).min()
            .expect("elements can not be empty");
        for el in &mut elements {
            el.borrow_mut().update_statistic(next_t, current_t);
        }
        current_t = next_t;
        for el in &mut elements {
            el.borrow_mut().set_current_t(current_t)
        }
        for el in &mut elements {
            if el.borrow().get_next_t() == current_t {
                el.borrow_mut().out_act();
            }
        }
        for el in &mut *elements {
            println!("{:?}", &*el.borrow_mut());
        }
    }
    for el in &elements {
        el.borrow().print_stats();
    }
}


fn main() {
    let delay_gen = DelayGen::Uniform(rand_distr::Uniform::<f64>::new(0., 10.));
    let element_process_3 = Rc::new(RefCell::new(ElementProcess::new(
        ElementBase::new("Process 3", delay_gen, Default::default()), 3,
        vec![Device::new("Device 1", delay_gen)]
    )));
    let element_process_2 = Rc::new(RefCell::new(ElementProcess::new(
        ElementBase::new(
            "Process 2", delay_gen,
            ProbabilityElementsMap::new(
                vec![(element_process_3.clone(), 1.0)]
            )
        ), 3,
        vec![
            Device::new("Device 1", delay_gen),
            // Device::new("Device 2", delay_gen),
            // Device::new("Device 3", delay_gen),
        ]
    )));
    let element_process_1 = Rc::new(RefCell::new(ElementProcess::new(
        ElementBase::new("Process 1", delay_gen,
            ProbabilityElementsMap::new(vec![(element_process_2.clone(), 1.0)])
        ),
        3,
        vec![
            Device::new("Device 1", delay_gen),
            // Device::new("Device 5", delay_gen),
            // Device::new("Device 6", delay_gen),
        ]
    )));

    let element_create = Rc::new(
        RefCell::new(
            ElementCreate(
                ElementBase::new(
                    "Create", delay_gen,
                    ProbabilityElementsMap::new(vec![(element_process_1.clone(), 1.0)])
                )
            )
        )
    );

    let elements: Vec<Rc<RefCell<dyn Element>>> = vec![
        element_create,
        element_process_1,
        element_process_2,
        element_process_3,
    ];

    simulate_model(elements, 500);
}