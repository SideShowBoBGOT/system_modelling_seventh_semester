use std::cell::RefCell;
use std::collections::BinaryHeap;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::distributions::Uniform;
use lab_4::{Event, Node, NodeBase, TimePoint};
use lab_4::delay_gen::DelayGen;
use lab_4::node_create::NodeCreate;
use lab_4::node_process::NodeProcess;
use lab_4::payload_queue::PayloadQueue;
use lab_4::prob_arr::{Probability, ProbabilityArray};
use lab_4::queue_resource::QueueResource;

fn simulate_model(mut events: BinaryHeap<Event>, end_time: TimePoint) {
    loop {
        let next_event = events.pop().unwrap();
        if next_event.get_current_t() > end_time {
            break next_event;
        }

        match next_event {
            Event::Create(event) => {
                let (event_self, next_event) = event.iterate();
                events.push(Event::Create(event_self));
                if let Some(next_event) = next_event {
                    events.push(next_event)
                }
            },
            Event::Process(event) => {
                let (event_self, next_event) = event.iterate();
                if let Some(event_self) = event_self {
                    events.push(Event::Process(event_self))
                }
                if let Some(next_event) = next_event {
                    events.push(next_event)
                }
            }
        }
    };
}

fn generate_recursive_sequential(depth: usize) -> (Node, Probability) {
    let prob_vec = if depth == 0 {
        ProbabilityArray::<Node>::new(
            vec![generate_recursive_sequential(depth - 1)]
        )
    } else {
        Default::default()
    };
    (
        Node::Process(RefCell::new(NodeProcess::new(
            NodeBase::new(prob_vec, DelayGen::Uniform(Uniform::new(5.0, 15.0))),
            QueueResource::new(PayloadQueue::default(), 10)
        ))),
        Probability::new(1.0)
    )
}

pub fn model_sequential(c: &mut Criterion) {
    let tree = NodeCreate::new(
        NodeBase::new(
            ProbabilityArray::<Node>::new(vec![generate_recursive_sequential(50)]),
            DelayGen::Uniform(Uniform::new(3.0, 7.0))
        )
    );

    let mut events = BinaryHeap::<Event>::new();
    events.push(Event::Create(tree.produce_event(TimePoint(0.0))));

    let mut group = c.benchmark_group("model_sequential");
    for size in (10000..=100000).step_by(10000) {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, i| b.iter(|| simulate_model(events.clone(), TimePoint(*i as f64))) );
    }
    group.finish();

}

pub fn model_trees_depth_2(c: &mut Criterion) {
    let prob_array = {
        let prob_array = {
            let node_prob = (
                Node::Process(RefCell::new(NodeProcess::new(
                    NodeBase::new(
                        Default::default(),
                        DelayGen::Uniform(Uniform::new(5.0, 15.0))
                    ),
                    QueueResource::new(
                        PayloadQueue::default(),
                        3
                    )
                ))),
                Probability::new(0.5)
            );
            ProbabilityArray::<Node>::new(vec![node_prob.clone(), node_prob])
        };
        let node = (
            Node::Process(RefCell::new(NodeProcess::new(
                NodeBase::new(
                    prob_array,
                    DelayGen::Uniform(Uniform::new(5.0, 15.0))
                ),
                QueueResource::new(
                    PayloadQueue::default(),
                    3
                )
            ))),
            Probability::new(0.1)
        );
        ProbabilityArray::<Node>::new(
            (0..10).map(|_| node.clone()).collect::<Vec<(Node, Probability)>>()
        )
    };

    let tree = NodeCreate::new(
        NodeBase::new(prob_array, DelayGen::Uniform(Uniform::new(10.0, 20.0)))
    );

    let mut events = BinaryHeap::<Event>::new();
    events.push(Event::Create(tree.produce_event(TimePoint(0.0))));

    let mut group = c.benchmark_group("model_trees_depth_2");
    for size in (10000..=100000).step_by(10000) {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, i| b.iter(|| simulate_model(events.clone(), TimePoint(*i as f64))) );
    }
    group.finish();
}

criterion_group!(benches, model_sequential, model_trees_depth_2);
criterion_main!(benches);
