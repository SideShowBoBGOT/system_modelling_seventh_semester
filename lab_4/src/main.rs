use std::cell::RefCell;
use std::ops::{Add, Sub};
use crate::delay_gen::DelayGen;
use crate::event_create::EventCreate;
use crate::event_process::EventProcess;
use crate::node_create::NodeCreate;
use crate::node_process::NodeProcess;
use crate::payload_queue::PayloadQueue;
use crate::prob_arr::ProbabilityArray;
use crate::queue_resource::{Queue, QueueProcessor};

#[derive(Debug, Copy, Clone, Default, PartialOrd, PartialEq)]
pub struct TimePoint(f64);

impl Sub for TimePoint {
    type Output = TimeSpan;
    fn sub(self, rhs: TimePoint) -> TimeSpan {
        TimeSpan(self.0 - rhs.0)
    }
}

impl Add<TimeSpan> for TimePoint {
    type Output = TimePoint;
    fn add(self, rhs: TimeSpan) -> TimePoint {
        TimePoint(self.0 + rhs.0)
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct TimeSpan(f64);

pub mod delay_gen {
    use rand::distributions::{Distribution};
    use crate::TimeSpan;
    use rand::thread_rng;
    use rand_distr::{Exp, Normal, Uniform};

    #[derive(Clone, Copy, Debug)]
    pub enum DelayGen {
        Normal(Normal<f64>),
        Uniform(Uniform<f64>),
        Exponential(Exp<f64>),
    }

    impl DelayGen {
        pub fn sample(&self) -> TimeSpan {
            TimeSpan(
                match self {
                    Self::Normal(dist) => dist.sample(&mut thread_rng()),
                    Self::Uniform(dist) => dist.sample(&mut thread_rng()),
                    Self::Exponential(dist) => dist.sample(&mut thread_rng()),
                }
            )
        }
    }
}

mod queue_resource {
    use std::cell::{Cell};
    use std::rc::{Rc, Weak};

    pub trait Queue {
        type Item;
        fn push(&mut self, t: Self::Item);
        fn pop(&mut self) -> Option<Self::Item>;
        fn is_empty(&self) -> bool;
    }

    #[derive(Debug, Default)]
    pub struct QueueResource<Q> {
        max_acquires: usize,
        acquires_count: Rc<Cell<usize>>,
        queue: Q,
    }

    pub struct QueueProcessor<E> {
        acquires_count: Weak<Cell<usize>>,
        value: E
    }

    impl<E> Drop for QueueProcessor<E> {
        fn drop(&mut self) {
            let acquires_count = self.acquires_count.upgrade().expect("Queue resource does not exist");
            acquires_count.set(acquires_count.get() - 1);
        }
    }

    impl<E> QueueProcessor<E> {
        pub fn value(&self) -> &E {
            &self.value
        }
        pub fn value_mut(&mut self) -> &mut E {
            &mut self.value
        }
    }

    impl<Q: Queue> QueueResource<Q> {
        pub fn new(queue: Q, max_acquires: usize) -> Self {
            Self{max_acquires, acquires_count: Rc::new(Cell::new(0usize)), queue}
        }

        pub fn push(&mut self, t: Q::Item) {
            self.queue.push(t)
        }

        pub fn is_empty(&self) -> bool {
            self.queue.is_empty()
        }

        pub fn is_any_free_processor(&self) -> bool {
            self.acquires_count.get() < self.max_acquires
        }

        pub fn acquire_processor(&mut self) -> QueueProcessor<Q::Item> {
            assert!(self.acquires_count.get() < self.max_acquires);
            let value = self.queue.pop().expect("Queue is empty");
            self.acquires_count.set(self.acquires_count.get() + 1);
            QueueProcessor{acquires_count: Rc::downgrade(&self.acquires_count), value}
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{Queue, QueueResource};

        #[derive(Default)]
        struct DummyQueue {
            len: usize
        }

        impl Queue for DummyQueue {
            type Item = ();
            fn push(&mut self, _: ()) {
                self.len += 1;
            }

            fn pop(&mut self) -> Option<()> {
                assert_eq!(self.is_empty(), false);
                self.len -= 1;
                Some(())
            }

            fn is_empty(&self) -> bool {
                self.len == 0
            }
        }

        #[test]
        fn test_one() {
            let mut res = QueueResource::new(DummyQueue::default(), 3usize);
            res.push(());
            res.push(());
            res.push(());
            res.push(());
            let _proc_one = res.acquire_processor();
            let _proc_two = res.acquire_processor();
            let _proc_three = res.acquire_processor();
        }

        #[test]
        #[should_panic]
        fn test_two() {
            let mut res = QueueResource::new(DummyQueue::default(), 2usize);
            res.push(());
            res.push(());
            res.push(());
            res.push(());

            let _proc_one = res.acquire_processor();
            let _proc_two = res.acquire_processor();
            let _proc_three = res.acquire_processor();
        }

        #[test]
        fn test_three() {
            let mut res = QueueResource::new(DummyQueue::default(), 2usize);
            res.push(());
            res.push(());
            res.push(());
            res.push(());

            let _proc_one = res.acquire_processor();
            let _proc_two = res.acquire_processor();
            drop(_proc_two);
            let _proc_three = res.acquire_processor();
        }

    }
}

pub mod prob_arr {
    use rand::random;

    pub(super) struct Probability(f64);
    impl Probability {
        pub(super) fn new(prob: f64) -> Self {
            assert!(prob >= 0.0 && prob <= 1.0);
            Self(prob)
        }
    }

    #[derive(Default)]
    pub(super) struct ProbabilityArray<T>(Vec<(T, Probability)>);

    impl<T> ProbabilityArray<T> {
        pub(super) fn new(next_elements_map: Vec<(T, Probability)>) -> Self {
            const EPSILON: f64 = 0.001;
            let sum = next_elements_map.iter().map(|e| (*e).1.0).sum::<f64>();
            assert!((sum - 1.0).abs() < EPSILON);
            Self(next_elements_map)
        }

        pub(super) fn sample(&self) -> Option<&T> {
            if self.0.is_empty() {
                return None;
            }

            let rand_value = random::<f64>();
            let mut current_sum = 0.0;

            let mut target_index = self.0.len() - 1;
            for (index, (_, prob)) in self.0.iter().enumerate() {
                current_sum += prob.0;
                if rand_value < current_sum {
                    target_index = index;
                    break;
                }
            }
            Some(&self.0.iter().nth(target_index)?.0)
        }
    }
}

#[derive(Clone, Default)]
struct Payload();

mod event_create {
    use crate::node_create::NodeCreate;
    use crate::{Event, Node, Payload, TimePoint};
    use crate::node_process::NodeProcess;

    pub(super) struct EventCreate {
        current_t: TimePoint,
        node: *const NodeCreate
    }

    impl EventCreate {
        pub(super) fn new(current_t: TimePoint, node: *const NodeCreate) -> Self {
            Self{current_t, node}
        }

        pub(super) fn iterate(self) -> (Self, Option<Event>) {
            let node = unsafe { &*self.node };
            let next_event = if let Some(next_node) = node.next_node() {
                match next_node {
                    Node::Create(node_create) => {
                        Some(Event::Create(node_create.produce_event(self.current_t)))
                    },
                    Node::Process(node_process) => {
                        NodeProcess::push_produce_event(node_process, self.current_t, Payload()).map(Event::Process)
                    }
                }
            } else {
                None
            };
            (node.produce_event(self.current_t), next_event)
        }
    }

}

mod event_process {
    use std::cell::RefCell;
    use crate::{Event, Node, Payload, TimePoint};
    use crate::node_process::NodeProcess;
    use crate::queue_resource::QueueProcessor;

    pub(super) struct EventProcess {
        current_t: TimePoint,
        node: *const RefCell<NodeProcess>,
        queue_processor: QueueProcessor<Payload>
    }

    impl EventProcess {
        pub(super) fn new(
            current_t: TimePoint,
            node: *const RefCell<NodeProcess>,
            queue_processor: QueueProcessor<Payload>
        ) -> Self {
            Self{current_t, node, queue_processor}
        }

        pub(super) fn iterate(mut self) -> (Option<Self>, Option<Event>) {
            let node = unsafe { &*self.node };
            let payload = std::mem::take(self.queue_processor.value_mut());
            drop(self.queue_processor);

            let next_event = {
                let node = node.borrow();
                if let Some(next_node) = node.next_node() {
                    match next_node {
                        Node::Create(node_create) => {
                            Some(Event::Create(node_create.produce_event(self.current_t)))
                        },
                        Node::Process(node_process) => {
                            NodeProcess::push_produce_event(node_process, self.current_t, payload).map(Event::Process)
                        }
                    }
                } else {
                    None
                }
            };
            (NodeProcess::pop_produce_event(node, self.current_t), next_event)
        }
    }
}


enum Event {
    Create(EventCreate),
    Process(EventProcess),
}

struct NodeBase {
    next_nodes: ProbabilityArray<Node>,
    delay_gen: DelayGen
}

mod node_create {
    use crate::{EventCreate, Node, NodeBase, TimePoint};

    pub(super) struct NodeCreate(NodeBase);

    impl NodeCreate {
        pub(super) fn next_node(&self) -> Option<&Node> {
            self.0.next_nodes.sample()
        }

        pub(super) fn produce_event(&self, old_t: TimePoint) -> EventCreate {
            EventCreate::new(old_t + self.0.delay_gen.sample(), self)
        }
    }
}



mod payload_queue {
    pub(super) struct PayloadQueue{
        pub(super) len: usize,
    }
}

impl Queue for PayloadQueue {
    type Item = Payload;

    fn push(&mut self, _: Self::Item) {
        self.len += 1;
    }

    fn pop(&mut self) -> Option<Self::Item> {
        if self.is_empty() {
            None
        } else {
            self.len -= 1;
            Some(Payload())
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

mod node_process {
    use std::cell::RefCell;
    use crate::{EventProcess, Node, NodeBase, Payload, PayloadQueue, TimePoint};
    use crate::queue_resource::QueueResource;

    pub(super) struct NodeProcess {
        base: NodeBase,
        queue: QueueResource<PayloadQueue>
    }

    impl NodeProcess {
        pub(super) fn new(base: NodeBase, queue: QueueResource<PayloadQueue>) -> Self {
            Self { base, queue }
        }

        pub(super) fn next_node(&self) -> Option<&Node> {
            self.base.next_nodes.sample()
        }

        pub(super) fn pop_produce_event(node: &RefCell<Self>, old_t: TimePoint) -> Option<EventProcess> {
            let is_any_free_processor = node.borrow().queue.is_any_free_processor();
            if is_any_free_processor {
                let delay = node.borrow().base.delay_gen.sample();
                Some(EventProcess::new(old_t + delay, node, node.borrow_mut().queue.acquire_processor()))
            } else {
                None
            }
        }

        pub(super)  fn push_produce_event(node: &RefCell<Self>, old_t: TimePoint, payload: Payload) -> Option<EventProcess> {
            node.borrow_mut().queue.push(payload);
            NodeProcess::pop_produce_event(node, old_t)
        }
    }
}


enum Node {
    Create(NodeCreate),
    Process(RefCell<NodeProcess>),
}

fn main() {



    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one() {
    }
}
