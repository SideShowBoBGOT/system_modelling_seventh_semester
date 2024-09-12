use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use rand::{thread_rng, RngCore};
use rand::distributions::{Distribution, Uniform};
use crate::task_2::create_patient::EventNewPatient;
use crate::task_2::event_terminal::EventTerminal;
use crate::task_2::patient::{Patient, PatientType};
use crate::task_2::queue_resource::{Queue, QueueProcessor, QueueResource};
use crate::task_2::transition_lab_reception::{
    EventTransitionFromLabToPatientWards,
    EventTransitionFromReceptionToLaboratory,
    EventTransition
};
use crate::{TimePoint, TimeSpan};

mod patient {
    use std::cmp::Ordering;
    use crate::{TimePoint};

    #[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Default)]
    pub enum PatientType {
        #[default]
        Three,
        Two,
        One,
    }

    #[derive(Debug, Default, Clone, Copy)]
    pub struct Patient {
        clinic_in_t: TimePoint,
        group: PatientType,
    }

    impl Patient {
        pub fn new(clinic_in_t: TimePoint, group: PatientType) -> Patient {
            Patient { clinic_in_t, group }
        }

        pub fn upgrade_to_first_group(mut self) -> Patient {
            self.group = PatientType::One;
            self
        }

        pub fn get_group(&self) -> PatientType {
            self.group
        }

        pub fn get_clinic_in_t(&self) -> TimePoint {
            self.clinic_in_t
        }
    }

    impl Eq for Patient {}

    impl PartialEq<Self> for Patient {
        fn eq(&self, other: &Self) -> bool {
            self.group.eq(&other.group) && self.clinic_in_t.eq(&other.clinic_in_t)
        }
    }

    impl PartialOrd<Self> for Patient {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            match self.group.cmp(&other.group) {
                Ordering::Equal => other.clinic_in_t.partial_cmp(&self.clinic_in_t),
                other => Some(other),
            }
        }
    }

    impl Ord for Patient {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(&other).expect("Patient ordering went wrong")
        }
    }

    #[cfg(test)]
    mod tests {
        use std::collections::BinaryHeap;
        use crate::task_2::patient::{Patient, PatientType};
        use crate::TimePoint;

        #[test]
        fn test_patient_type_ordering() {
            let mut bh = BinaryHeap::new();
            bh.push(PatientType::Three);
            bh.push(PatientType::Two);
            bh.push(PatientType::One);

            assert_eq!(bh.pop(), Some(PatientType::One));
            assert_eq!(bh.pop(), Some(PatientType::Two));
            assert_eq!(bh.pop(), Some(PatientType::Three));
        }

        #[test]
        fn test_patient_ordering() {
            let mut bh = BinaryHeap::new();

            bh.push(Patient::new(TimePoint(100.0), PatientType::Three));
            bh.push(Patient::new(TimePoint(50.0), PatientType::Three));
            bh.push(Patient::new(TimePoint(25.0), PatientType::Three));

            bh.push(Patient::new(TimePoint(900.0), PatientType::Two));
            bh.push(Patient::new(TimePoint(800.0), PatientType::Two));
            bh.push(Patient::new(TimePoint(700.0), PatientType::Two));

            bh.push(Patient::new(TimePoint(2299.0), PatientType::One));
            bh.push(Patient::new(TimePoint(999.0), PatientType::One));
            bh.push(Patient::new(TimePoint(1999.0), PatientType::One));

            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(999.0), PatientType::One)));
            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(1999.0), PatientType::One)));
            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(2299.0), PatientType::One)));

            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(700.0), PatientType::Two)));
            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(800.0), PatientType::Two)));
            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(900.0), PatientType::Two)));

            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(25.0), PatientType::Three)));
            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(50.0), PatientType::Three)));
            assert_eq!(bh.pop(), Some(Patient::new(TimePoint(100.0), PatientType::Three)));
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
        use crate::task_2::queue_resource::{Queue, QueueResource};

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

impl<T: Ord> Queue for BinaryHeap<T> {
    type Item = T;

    fn push(&mut self, t: Self::Item) {
        BinaryHeap::push(self, t)
    }

    fn pop(&mut self) -> Option<Self::Item> {
        BinaryHeap::pop(self)
    }

    fn is_empty(&self) -> bool {
        BinaryHeap::is_empty(self)
    }
}

impl<T> Queue for VecDeque<T> {
    type Item = T;

    fn push(&mut self, t: Self::Item) {
        self.push_back(t);
    }

    fn pop(&mut self) -> Option<Self::Item> {
        self.pop_front()
    }

    fn is_empty(&self) -> bool {
        VecDeque::is_empty(self)
    }
}

#[derive(Debug)]
struct ReceptionDepartment(QueueResource<BinaryHeap<Patient>>);

#[derive(Debug)]
struct PatientWards(QueueResource<VecDeque<Patient>>);

#[derive(Debug)]
struct LabRegistry(QueueResource<VecDeque<Patient>>);

#[derive(Debug)]
struct Laboratory(QueueResource<VecDeque<Patient>>);

mod create_patient {
    use lazy_static::lazy_static;
    use rand::distributions::{Distribution, Uniform};
    use rand_distr::Exp;
    use crate::task_2::{EventReceptionDepartment, Patient, ReceptionDepartment};
    use crate::{TimePoint, TimeSpan};
    use crate::task_2::patient::PatientType;

    #[derive(Debug, Default)]
    pub struct EventNewPatient {
        current_t: TimePoint,
        patient: Patient,
    }

    lazy_static! {
        static ref DELAY_GEN: Exp<f64> = Exp::new(1.0 / 15.0).expect("Failed to create delay gen");
    }

    fn generate_patient_type() -> PatientType {
        let value = Uniform::new(0.0, 1.0).sample(&mut rand::thread_rng());
        match value {
            ..0.5 => PatientType::One,
            0.5..0.6 => PatientType::Two,
            0.6.. => PatientType::Three,
            _ => panic!("PatientType::generate_patient error")
        }
    }

    impl EventNewPatient {

        pub fn get_current_t(&self) -> TimePoint {
            self.current_t
        }

        pub fn iterate(self, reception_department: &mut ReceptionDepartment) -> (Self, Option<EventReceptionDepartment>) {
            reception_department.0.push(self.patient);
            let delay = TimeSpan(DELAY_GEN.sample(&mut rand::thread_rng()));
            let patient_t = self.current_t + delay;
            (
                Self {
                    current_t: patient_t,
                    patient: Patient::new(patient_t, generate_patient_type()),
                },
                EventReceptionDepartment::new(self.current_t, reception_department)
            )
        }
    }
}

struct EventBase {
    current_t: TimePoint,
    patient_processor: QueueProcessor<Patient>
}

macro_rules! impl_event_new {
    ($event_type:ty, $queue_type:ty) => {
        impl $event_type {
            fn new(old_current_t: TimePoint, queue: &mut $queue_type) -> Option<Self> {
                if queue.0.is_any_free_processor() && !queue.0.is_empty() {
                    let patient_processor = queue.0.acquire_processor();
                    let current_t = old_current_t + Self::determine_delay(patient_processor.value().get_group());
                    Some(Self(EventBase{current_t, patient_processor}))
                } else {
                    None
                }
            }
        }
    };
}

struct EventReceptionDepartment(EventBase);
impl_event_new!(EventReceptionDepartment, ReceptionDepartment);

impl EventReceptionDepartment {
    fn determine_delay(patient_type: PatientType) -> TimeSpan {
        match patient_type {
            PatientType::One => TimeSpan(15.0),
            PatientType::Two => TimeSpan(40.0),
            PatientType::Three => TimeSpan(30.0),
        }
    }
}

enum ReceptionDepartmentTransitionToResult {
    PatientWards(Option<EventPatientWards>),
    FromReceptionToLaboratory(EventTransitionFromReceptionToLaboratory)
}

impl EventReceptionDepartment {
    pub fn iterate(
        self, reception_department: &mut ReceptionDepartment, patient_wards: &mut PatientWards
    ) -> (Option<Self>, ReceptionDepartmentTransitionToResult) {
        let transition_to = match self.0.patient_processor.value().get_group() {
            PatientType::One => ReceptionDepartmentTransitionToResult::PatientWards ({
                patient_wards.0.push(self.0.patient_processor.value().clone());
                EventPatientWards::new(self.0.current_t, patient_wards)
            }),
            PatientType::Two | PatientType::Three => {
                ReceptionDepartmentTransitionToResult::FromReceptionToLaboratory(EventTransitionFromReceptionToLaboratory(
                    EventTransition::new(self.0.current_t, self.0.patient_processor.value().clone())
                ))
            },
        };
        drop(self.0.patient_processor);
        (Self::new(self.0.current_t, reception_department), transition_to)
    }
}

mod transition_lab_reception {
    use lazy_static::lazy_static;
    use rand::distributions::{Distribution, Uniform};
    use crate::task_2::{EventLabRegistration, EventPatientWards, LabRegistry, Patient, PatientWards};
    use crate::{TimePoint, TimeSpan};

    lazy_static! {
        static ref RECEPTION_LABORATORY_TRANSITION_DELAY: Uniform<f64> = Uniform::new(2.0, 5.0);
    }

    pub struct EventTransition {
        current_t: TimePoint,
        patient: Patient,
    }

    impl EventTransition {
        pub fn new(old_current_t: TimePoint, patient: Patient) -> Self {
            let delay = TimeSpan(RECEPTION_LABORATORY_TRANSITION_DELAY.sample(&mut rand::thread_rng()));
            Self{current_t: old_current_t + delay, patient}
        }
    }

    pub struct EventTransitionFromReceptionToLaboratory(pub EventTransition);

    impl EventTransitionFromReceptionToLaboratory {
        pub fn get_current_t(&self) -> TimePoint {
            self.0.current_t
        }

        pub fn iterate(self, lab_registry: &mut LabRegistry) -> Option<EventLabRegistration> {
            lab_registry.0.push(self.0.patient);
            EventLabRegistration::new(self.0.current_t, lab_registry)
        }
    }

    pub struct EventTransitionFromLabToPatientWards(pub EventTransition);

    impl EventTransitionFromLabToPatientWards {
        pub fn get_current_t(&self) -> TimePoint {
            self.0.current_t
        }

        pub(super) fn iterate(self, patient_wards: &mut PatientWards) -> Option<EventPatientWards> {
            patient_wards.0.push(self.0.patient);
            EventPatientWards::new(self.0.current_t, patient_wards)
        }
    }
}

struct EventPatientWards(EventBase);
impl_event_new!(EventPatientWards, PatientWards);
impl EventPatientWards {
    fn determine_delay(_: PatientType) -> TimeSpan {
        TimeSpan(Uniform::new(3.0, 8.0).sample(&mut thread_rng()))
    }

    pub fn iterate(self, patient_wards: &mut PatientWards) -> (Option<Self>, EventTerminal) {
        let next_event = EventTerminal::new(self.0.current_t, self.0.patient_processor.value().clone());
        drop(self.0.patient_processor);
        (Self::new(self.0.current_t, patient_wards), next_event)
    }
}

mod event_terminal {
    use crate::task_2::patient::Patient;
    use crate::TimePoint;

    pub struct EventTerminal {
        current_t: TimePoint,
        patient: Patient
    }

    impl EventTerminal {
        pub fn new(current_t: TimePoint, patient: Patient) -> Self {
            Self{current_t, patient}
        }
        pub fn get_current_t(&self) -> TimePoint {
            self.current_t
        }
        pub fn get_patient(&self) -> &Patient {
            &self.patient
        }
    }
}

fn get_erlang_distribution(shape: i64, scale: f64) -> rand_simple::Erlang {
    let mut erlang = rand_simple::Erlang::new(
        [
            thread_rng().next_u32(),
            thread_rng().next_u32(),
            thread_rng().next_u32(),
        ]
    );
    erlang.try_set_params(shape, scale).expect("Erlang set params failed");
    erlang
}

struct EventLabRegistration(EventBase);
impl_event_new!(EventLabRegistration, LabRegistry);
impl EventLabRegistration {
    fn determine_delay(_: PatientType) -> TimeSpan {
        let delay = TimeSpan(get_erlang_distribution(3, 4.5).sample());
        // println!("{:?}", delay);
        delay
    }
    pub fn iterate(self, lab_registry: &mut LabRegistry, laboratory: &mut Laboratory)
        -> (Option<Self>, Option<EventLaboratory>) {
        laboratory.0.push(self.0.patient_processor.value().clone());
        drop(self.0.patient_processor);
        (Self::new(self.0.current_t, lab_registry), EventLaboratory::new(self.0.current_t, laboratory))
    }
}

struct EventLaboratory(EventBase);
impl_event_new!(EventLaboratory, Laboratory);
pub enum EventLaboratoryTransitionResult {
    TransitionFromLabToPatientWards(EventTransitionFromLabToPatientWards),
    Terminal(EventTerminal)
}

impl EventLaboratory {
    fn determine_delay(_: PatientType) -> TimeSpan {
        let delay = TimeSpan(get_erlang_distribution(2, 4.0).sample());
        delay
    }
    pub fn iterate(self, laboratory: &mut Laboratory) -> (Option<Self>, EventLaboratoryTransitionResult) {
        let transition_to = match self.0.patient_processor.value().get_group() {
            PatientType::One => panic!("Patient one can not be in the laboratory"),
            PatientType::Two => EventLaboratoryTransitionResult::TransitionFromLabToPatientWards(
                EventTransitionFromLabToPatientWards(
                    EventTransition::new(
                        self.0.current_t, self.0.patient_processor.value().clone().upgrade_to_first_group()
                    )
                )
            ),
            PatientType::Three => EventLaboratoryTransitionResult::Terminal(
                EventTerminal::new(self.0.current_t, self.0.patient_processor.value().clone())
            ),
        };
        drop(self.0.patient_processor);
        (Self::new(self.0.current_t, laboratory), transition_to)
    }
}

enum Event {
    NewPatient(EventNewPatient),
    ReceptionDepartment(EventReceptionDepartment),
    FromReceptionLaboratory(EventTransitionFromReceptionToLaboratory),
    FromLabToReception(EventTransitionFromLabToPatientWards),
    PatientWards(EventPatientWards),
    LabRegistration(EventLabRegistration),
    Laboratory(EventLaboratory),
    Terminal(EventTerminal),
}

impl Event {
    fn get_current_t(&self) -> TimePoint {
        match self {
            Event::NewPatient(event) => event.get_current_t(),
            Event::ReceptionDepartment(event) => event.0.current_t,
            Event::FromReceptionLaboratory(event) => event.get_current_t(),
            Event::FromLabToReception(event) => event.get_current_t(),
            Event::PatientWards(event) => event.0.current_t,
            Event::LabRegistration(event) => event.0.current_t,
            Event::Laboratory(event) => event.0.current_t,
            Event::Terminal(event) => event.get_current_t(),
        }
    }
}

impl Default for Event {
    fn default() -> Self {
        Self::NewPatient(EventNewPatient::default())
    }
}

impl Eq for Event {}

impl PartialEq<Self> for Event {
    fn eq(&self, other: &Self) -> bool {
        self.get_current_t() == other.get_current_t()
    }
}

impl PartialOrd<Self> for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.get_current_t().partial_cmp(&self.get_current_t())
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).expect("Failed to compare events")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BinaryHeap, VecDeque};
    use crate::task_2::{
        Event, EventLaboratoryTransitionResult, LabRegistry,
        Laboratory, PatientWards, ReceptionDepartment, ReceptionDepartmentTransitionToResult
    };
    use crate::task_2::event_terminal::EventTerminal;
    use crate::task_2::queue_resource::QueueResource;
    use crate::{TimePoint, TimeSpan};
    use crate::task_2::patient::Patient;

    #[test]
    fn test_general() {
        let end_time = TimePoint(100000.0);

        let mut reception_department = ReceptionDepartment(QueueResource::new(BinaryHeap::new(), 2));
        let mut patient_wards = PatientWards(QueueResource::new(VecDeque::new(), 3));
        let mut lab_registry = LabRegistry(QueueResource::new(VecDeque::new(), 1));
        let mut laboratory = Laboratory(QueueResource::new(VecDeque::new(), 2));

        let mut last_lab_registration = TimePoint(0.0);
        let mut lab_registration_interval_sum = TimeSpan(0.0);
        let mut lab_registration_count = 0usize;

        let mut total_terminal_patients = 0usize;
        let mut total_patients_spent_time = TimeSpan::default();

        let mut nodes = BinaryHeap::new();
        nodes.push(Event::default());

        let _ = loop {

            let next_event = nodes.pop().unwrap();
            if next_event.get_current_t() > end_time {
                break next_event;
            }
            match next_event {
                Event::NewPatient(event) => {
                    let res = event.iterate(&mut reception_department);
                    nodes.push(Event::NewPatient(res.0));
                    if let Some(next_event) = res.1 {
                        nodes.push(Event::ReceptionDepartment(next_event));
                    }
                },
                Event::ReceptionDepartment(event) => {
                    let res = event.iterate(&mut reception_department, &mut patient_wards);
                    if let Some(self_event) = res.0 {
                        nodes.push(Event::ReceptionDepartment(self_event));
                    }
                    match res.1 {
                        ReceptionDepartmentTransitionToResult::PatientWards(event) => {
                            if let Some(event) = event {
                                nodes.push(Event::PatientWards(event));
                            }
                        }
                        ReceptionDepartmentTransitionToResult::FromReceptionToLaboratory(event) => {
                            nodes.push(Event::FromReceptionLaboratory(event));
                        }
                    }
                },
                Event::FromReceptionLaboratory(event) => {
                    if let Some(res) = event.iterate(&mut lab_registry) {
                        nodes.push(Event::LabRegistration(res));
                    }
                },
                Event::FromLabToReception(event) => {
                    if let Some(res) = event.iterate(&mut patient_wards) {
                        nodes.push(Event::PatientWards(res));
                    }
                },
                Event::PatientWards(event) => {
                    let (self_event, terminal) = event.iterate(&mut patient_wards);
                    if let Some(res_event) = self_event {
                        nodes.push(Event::PatientWards(res_event));
                    }
                    nodes.push(Event::Terminal(terminal));
                },
                Event::LabRegistration(event) => {
                    lab_registration_count += 1;
                    lab_registration_interval_sum += event.0.current_t - last_lab_registration;
                    last_lab_registration = event.0.current_t;

                    let (self_event, next_event) = event.iterate(&mut lab_registry, &mut laboratory);
                    if let Some(self_event) = self_event {
                        nodes.push(Event::LabRegistration(self_event));
                    }
                    if let Some(next_event) = next_event {
                        nodes.push(Event::Laboratory(next_event));
                    }
                },
                Event::Laboratory(event) => {
                    let (self_event, next_event) = event.iterate(&mut laboratory);
                    if let Some(self_event) = self_event {
                        nodes.push(Event::Laboratory(self_event));
                    }
                    match next_event {
                        EventLaboratoryTransitionResult::TransitionFromLabToPatientWards(next_event) => {
                            nodes.push(Event::FromLabToReception(next_event));
                        }
                        EventLaboratoryTransitionResult::Terminal(event) => {
                            nodes.push(Event::Terminal(event));
                        }
                    }
                },
                Event::Terminal(event) => {
                    total_terminal_patients += 1;
                    total_patients_spent_time += event.get_current_t() - event.get_patient().get_clinic_in_t();
                },
            }
        };

        println!("Mean time: {:?}", total_patients_spent_time.0 / total_terminal_patients as f64);
        println!("Lab registration mean time: {:?}", lab_registration_interval_sum.0 / lab_registration_count as f64);
    }

    #[test]
    fn test_event_ordering() {
        let mut bh = BinaryHeap::new();

        bh.push(Event::Terminal(EventTerminal::new(TimePoint(10.0), Patient::default())));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(9.0), Patient::default())));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(8.0), Patient::default())));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(14.0), Patient::default())));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(1.0), Patient::default())));

        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(1.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(8.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(9.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(10.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(14.0));
    }
}