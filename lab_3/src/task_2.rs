use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use rand::{thread_rng, RngCore};
use crate::task_2::create_patient::EventNewPatient;
use crate::task_2::event_lab_registration::EventLabRegistration;
use crate::task_2::event_laboratory::EventLaboratory;
use crate::task_2::event_patient_wards::EventPatientWards;
use crate::task_2::event_reception_department::EventReceptionDepartment;
use crate::task_2::event_terminal::EventTerminal;
use crate::task_2::patient::Patient;
use crate::task_2::transition_lab_reception::{EventTransitionFromLabToReception, EventTransitionFromReceptionToLaboratory};
use crate::TimePoint;

mod patient {
    use std::cmp::Ordering;
    use crate::{TimePoint, TimeSpan};

    #[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Default)]
    pub enum PatientType {
        #[default]
        Three,
        Two,
        One,
    }

    #[derive(Debug, Default, Copy, Clone)]
    pub struct Patient {
        clinic_in_t: TimePoint,
        group: PatientType,
        last_update_time: TimePoint,
        time_spent: TimeSpan
    }

    impl Patient {
        pub fn new(clinic_in_t: TimePoint, group: PatientType) -> Patient {
            Patient {
                clinic_in_t, group, last_update_time: clinic_in_t,
                time_spent: TimeSpan::default()
            }
        }

        pub fn upgrade_to_first_group(mut self) -> Patient {
            self.group = PatientType::One;
            self
        }

        pub fn update_time(&mut self, current_t: TimePoint) {
            assert!(current_t > self.last_update_time);
            self.time_spent += current_t - self.last_update_time;
            self.last_update_time = current_t;
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


// mod queue_resource {
//     use std::cell::{Cell};
//     use std::rc::{Rc, Weak};
//
//     pub trait Queue {
//         type Item;
//         fn push(&mut self, t: Self::Item);
//         fn pop(&mut self) -> Option<Self::Item>;
//         fn is_empty(&self) -> bool;
//     }
//
//     pub struct QueueResource<Q> {
//         max_acquires: usize,
//         acquires_count: Rc<Cell<usize>>,
//         queue: Q,
//     }
//
//     pub struct QueueProcessor<E> {
//         acquires_count: Weak<Cell<usize>>,
//         value: E
//     }
//
//     impl<E> Drop for QueueProcessor<E> {
//         fn drop(&mut self) {
//             let acquires_count = self.acquires_count.upgrade().expect("Queue resource does not exist");
//             acquires_count.set(acquires_count.get() - 1);
//         }
//     }
//
//     impl<E> QueueProcessor<E> {
//         pub fn value(&self) -> &E {
//             &self.value
//         }
//
//         pub fn value_mut(&mut self) -> &mut E {
//             &mut self.value
//         }
//     }
//
//     impl<Q: Queue> QueueResource<Q> {
//         pub fn new(queue: Q, max_acquires: usize) -> Self {
//             Self{max_acquires, acquires_count: Rc::new(Cell::new(0usize)), queue}
//         }
//
//         pub fn push(&mut self, t: Q::Item) {
//             self.queue.push(t)
//         }
//
//         pub fn is_empty(&self) -> bool {
//             self.queue.is_empty()
//         }
//
//         pub fn acquire_processor(&mut self) -> QueueProcessor<Q::Item> {
//             assert!(self.acquires_count.get() < self.max_acquires);
//             let value = self.queue.pop().expect("Queue is empty");
//             self.acquires_count.set(self.acquires_count.get() + 1);
//             QueueProcessor{acquires_count: Rc::downgrade(&self.acquires_count), value}
//         }
//     }
//
//     #[cfg(test)]
//     mod tests {
//         use crate::task_2::queue_resource::{Queue, QueueResource};
//
//         #[derive(Default)]
//         struct DummyQueue {
//             len: usize
//         }
//
//         impl Queue for DummyQueue {
//             type Item = ();
//             fn push(&mut self, t: ()) {
//                 self.len += 1;
//             }
//
//             fn pop(&mut self) -> Option<()> {
//                 assert_eq!(self.is_empty(), false);
//                 self.len -= 1;
//                 Some(())
//             }
//
//             fn is_empty(&self) -> bool {
//                 self.len == 0
//             }
//         }
//
//         #[test]
//         fn test_one() {
//             let mut res = QueueResource::new(DummyQueue::default(), 3usize);
//             res.push(());
//             res.push(());
//             res.push(());
//             res.push(());
//             let proc_one = res.acquire_processor();
//             let proc_two = res.acquire_processor();
//             let proc_three = res.acquire_processor();
//         }
//
//         #[test]
//         #[should_panic]
//         fn test_two() {
//             let mut res = QueueResource::new(DummyQueue::default(), 2usize);
//             res.push(());
//             res.push(());
//             res.push(());
//             res.push(());
//
//             let proc_one = res.acquire_processor();
//             let proc_two = res.acquire_processor();
//             let proc_three = res.acquire_processor();
//         }
//
//         #[test]
//         fn test_three() {
//             let mut res = QueueResource::new(DummyQueue::default(), 2usize);
//             res.push(());
//             res.push(());
//             res.push(());
//             res.push(());
//
//             let proc_one = res.acquire_processor();
//             let proc_two = res.acquire_processor();
//             drop(proc_two);
//             let proc_three = res.acquire_processor();
//         }
//
//     }
// }

#[derive(Debug, Default)]
struct Clinic {
    reception_department: ReceptionDepartment,
    patient_wards: PatientWards,
    lab_registry: LabRegistry,
    laboratory: Laboratory
}

#[derive(Debug, Default)]
struct ReceptionDepartment {
    queue: BinaryHeap<Patient>,
    is_doctor_busy: [bool; 2],
}

#[derive(Debug, Default)]
struct PatientWards {
    queue_size: usize,
    is_attendant_busy: [bool; 3],
}

#[derive(Debug, Default)]
struct LabRegistry {
    queue: VecDeque<Patient>,
    is_busy: bool
}

#[derive(Debug, Default)]
struct Laboratory {
    queue: VecDeque<Patient>,
    is_lab_assistant_busy: [bool; 2],
}

mod create_patient {
    use std::cell::RefCell;
    use std::rc::Rc;
    use lazy_static::lazy_static;
    use rand::distributions::{Distribution, Uniform};
    use rand_distr::Exp;
    use crate::task_2::event_reception_department::EventReceptionDepartment;
    use crate::task_2::{Clinic, Patient};
    use crate::{TimePoint, TimeSpan};
    use crate::task_2::patient::PatientType;

    #[derive(Debug, Default)]
    pub struct EventNewPatient {
        current_t: TimePoint,
        patient: Patient,
        clinic: Rc<RefCell<Clinic>>
    }

    lazy_static! {
        static ref DELAY_GEN: Exp<f64> = Exp::new(15.0).expect("Failed to create delay gen");
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

        pub fn iterate(self) -> (Self, Option<EventReceptionDepartment>) {

            let free_doctor_index = self.clinic.borrow()
                .reception_department.is_doctor_busy.iter().position(|d| !*d);
            let event_reception_dep = if let Some(index) = free_doctor_index {
                assert!(self.clinic.borrow_mut().reception_department.queue.is_empty());
                self.clinic.borrow_mut().reception_department.is_doctor_busy[index] = true;
                Some(EventReceptionDepartment::new(self.current_t, index, self.clinic.clone(), self.patient))
            } else {
                self.clinic.borrow_mut().reception_department.queue.push(self.patient);
                None
            };

            let patient_t = self.current_t + TimeSpan(DELAY_GEN.sample(&mut rand::thread_rng()));
            (
                Self {
                    current_t: patient_t,
                    patient: Patient::new(patient_t, generate_patient_type()),
                    clinic: self.clinic.clone()
                },
                event_reception_dep,
            )
        }
    }
}

mod event_reception_department {
    use std::cell::RefCell;
    use std::rc::Rc;
    use crate::task_2::{Clinic, Patient};
    use crate::task_2::event_patient_wards::EventPatientWards;
    use crate::task_2::transition_lab_reception::{EventTransitionFromReceptionToLaboratory, EventTransitionReceptionLaboratory};
    use crate::{TimePoint, TimeSpan};
    use crate::task_2::patient::PatientType;

    pub struct EventReceptionDepartment {
        current_t: TimePoint,
        doctor_index: usize,
        clinic: Rc<RefCell<Clinic>>,
        patient: Patient
    }

    fn determine_delay(patient_type: PatientType) -> TimeSpan {
        match patient_type {
            PatientType::One => TimeSpan(15.0),
            PatientType::Two => TimeSpan(40.0),
            PatientType::Three => TimeSpan(30.0),
        }
    }

    pub enum ReceptionDepartmentTransitionToResult {
        PatientWards(Option<EventPatientWards>),
        FromReceptionToLaboratory(EventTransitionFromReceptionToLaboratory)
    }

    impl EventReceptionDepartment {
        pub fn get_current_t(&self) -> TimePoint {
            self.current_t
        }

        pub fn new(old_current_t: TimePoint, doctor_index: usize, clinic: Rc<RefCell<Clinic>>, patient: Patient) -> Self {
            Self{current_t: old_current_t + determine_delay(patient.get_group()), doctor_index, clinic, patient}
        }

        pub fn iterate(self) -> (Option<EventReceptionDepartment>, ReceptionDepartmentTransitionToResult) {
            let transition_to = match self.patient.get_group() {
                PatientType::One => ReceptionDepartmentTransitionToResult::PatientWards (
                    {
                        let free_attendant_index = self.clinic.borrow()
                            .patient_wards.is_attendant_busy.iter().position(|d| !*d);
                        if let Some(index) = free_attendant_index {
                            assert_eq!(self.clinic.borrow().patient_wards.queue_size, 0);
                            self.clinic.borrow_mut().patient_wards.is_attendant_busy[index] = true;
                            Some(EventPatientWards::new(self.current_t, self.clinic.clone(), index))
                        } else {
                            self.clinic.borrow_mut().patient_wards.queue_size += 1;
                            None
                        }
                    }
                ),
                PatientType::Two | PatientType::Three => {
                    ReceptionDepartmentTransitionToResult::FromReceptionToLaboratory({
                        EventTransitionFromReceptionToLaboratory(
                            EventTransitionReceptionLaboratory::new(self.current_t, self.clinic.clone(), self.patient)
                        )
                    })
                },
            };
            let next_reception_dep = {
                let mut clinic = self.clinic.borrow_mut();
                if let Some(patient) = clinic.reception_department.queue.pop() {
                    Some(Self::new(self.current_t, self.doctor_index, self.clinic.clone(), patient))
                } else {
                    clinic.reception_department.is_doctor_busy[self.doctor_index] = false;
                    None
                }
            };
            (next_reception_dep, transition_to)
        }
    }
}


mod transition_lab_reception {
    use std::cell::RefCell;
    use std::rc::Rc;
    use lazy_static::lazy_static;
    use rand::distributions::{Distribution, Uniform};
    use crate::task_2::{Clinic, Patient};
    use crate::task_2::event_lab_registration::EventLabRegistration;
    use crate::task_2::event_reception_department::EventReceptionDepartment;
    use crate::{TimePoint, TimeSpan};

    lazy_static! {
        static ref RECEPTION_LABORATORY_TRANSITION_DELAY: Uniform<f64> = Uniform::new(2.0, 5.0);
    }

    pub struct EventTransitionReceptionLaboratory {
        current_t: TimePoint,
        clinic: Rc<RefCell<Clinic>>,
        patient: Patient,
    }

    impl EventTransitionReceptionLaboratory {
        pub fn new(old_current_t: TimePoint, clinic: Rc<RefCell<Clinic>>, patient: Patient) -> Self {
            let delay = TimeSpan(RECEPTION_LABORATORY_TRANSITION_DELAY.sample(&mut rand::thread_rng()));
            Self{current_t: old_current_t + delay, clinic, patient}
        }
    }

    pub struct EventTransitionFromReceptionToLaboratory(pub EventTransitionReceptionLaboratory);

    impl EventTransitionFromReceptionToLaboratory {
        pub fn get_current_t(&self) -> TimePoint {
            self.0.current_t
        }

        pub fn iterate(self) -> Option<EventLabRegistration> {
            if self.0.clinic.borrow().lab_registry.is_busy {
                self.0.clinic.borrow_mut().lab_registry.queue.push_back(self.0.patient);
                None
            } else {
                assert!(self.0.clinic.borrow().lab_registry.queue.is_empty());
                self.0.clinic.borrow_mut().lab_registry.is_busy = true;
                Some(EventLabRegistration::new(self.0.current_t, self.0.clinic, self.0.patient))
            }
        }
    }

    pub struct EventTransitionFromLabToReception(pub EventTransitionReceptionLaboratory);

    impl EventTransitionFromLabToReception {
        pub fn get_current_t(&self) -> TimePoint {
            self.0.current_t
        }

        pub fn iterate(self) -> Option<EventReceptionDepartment> {
            let free_doctor_index = self.0.clinic.borrow()
                .reception_department.is_doctor_busy.iter().position(|d| !*d);
            if let Some(index) = free_doctor_index {
                assert!(self.0.clinic.borrow_mut().reception_department.queue.is_empty());
                self.0.clinic.borrow_mut().reception_department.is_doctor_busy[index] = true;
                Some(EventReceptionDepartment::new(self.0.current_t, index, self.0.clinic.clone(), self.0.patient))
            } else {
                self.0.clinic.borrow_mut().reception_department.queue.push(self.0.patient);
                None
            }
        }
    }

}

mod event_patient_wards {
    use std::cell::RefCell;
    use std::rc::Rc;
    use lazy_static::lazy_static;
    use rand::distributions::{Distribution, Uniform};
    use crate::task_2::{Clinic};
    use crate::task_2::event_terminal::EventTerminal;
    use crate::{TimePoint, TimeSpan};

    pub struct EventPatientWards {
        current_t: TimePoint,
        clinic: Rc<RefCell<Clinic>>,
        attendant_index: usize
    }

    lazy_static! {
        static ref DELAY_GEN: Uniform<f64> = Uniform::new(3.0, 8.0);
    }

    impl EventPatientWards {
        pub fn get_current_t(&self) -> TimePoint {
            self.current_t
        }

        pub fn new(old_current_t: TimePoint, clinic: Rc<RefCell<Clinic>>, attendant_index: usize) -> Self {
            let delay = TimeSpan(DELAY_GEN.sample(&mut rand::thread_rng()));
            Self{current_t: old_current_t + delay, clinic, attendant_index}
        }

        pub fn iterate(self) -> (Option<Self>, EventTerminal) {
            let mut clinic = self.clinic.borrow_mut();
            let next_event = if clinic.patient_wards.queue_size > 0 {
                clinic.patient_wards.queue_size -= 1;
                Some(Self::new(self.current_t, self.clinic.clone(), self.attendant_index))
            } else {
                clinic.patient_wards.is_attendant_busy[self.attendant_index] = false;
                None
            };
            (next_event, EventTerminal::new(self.current_t))
        }
    }
}

mod event_terminal {
    use crate::TimePoint;

    pub struct EventTerminal {
        current_t: TimePoint,
    }

    impl EventTerminal {
        pub fn new(current_t: TimePoint) -> Self {
            Self{current_t}
        }
        pub fn get_current_t(&self) -> TimePoint {
            self.current_t
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

mod event_lab_registration {
    use std::cell::RefCell;
    use std::rc::Rc;
    use crate::task_2::{get_erlang_distribution, Clinic, Patient};
    use crate::task_2::event_laboratory::EventLaboratory;
    use crate::{TimePoint, TimeSpan};

    fn sample_delay() -> TimeSpan {
        TimeSpan(get_erlang_distribution(3, 4.5).sample())
    }

    pub struct EventLabRegistration {
        current_t: TimePoint,
        clinic: Rc<RefCell<Clinic>>,
        patient: Patient
    }

    impl EventLabRegistration {
        pub fn get_current_t(&self) -> TimePoint {
            self.current_t
        }

        pub fn new(
            old_current_t: TimePoint,
            clinic: Rc<RefCell<Clinic>>,
            patient: Patient
        ) -> Self {
            Self{current_t: old_current_t + sample_delay(), clinic, patient}
        }

        pub fn iterate(self) -> (Option<Self>, Option<EventLaboratory>) {
            let free_assistant_index = self.clinic.borrow()
                .laboratory.is_lab_assistant_busy.iter().position(|d| !*d);
            let event_lab = if let Some(index) = free_assistant_index {
                assert!(self.clinic.borrow().laboratory.queue.is_empty());
                self.clinic.borrow_mut().laboratory.is_lab_assistant_busy[index] = true;
                Some(EventLaboratory::new(self.current_t, self.clinic.clone(), self.patient, index))
            } else {
                self.clinic.borrow_mut().laboratory.queue.push_back(self.patient);
                None
            };
            let next_reg = {
                let mut clinic = self.clinic.borrow_mut();
                if let Some(patient) = clinic.lab_registry.queue.pop_front() {
                    Some(Self::new(self.current_t, self.clinic.clone(), patient))
                } else {
                    clinic.lab_registry.is_busy = false;
                    None
                }
            };
            (next_reg, event_lab)
        }
    }
}

mod event_laboratory {
    use std::cell::RefCell;
    use std::rc::Rc;
    use crate::task_2::{get_erlang_distribution, Clinic, Patient};
    use crate::task_2::event_terminal::EventTerminal;
    use crate::task_2::transition_lab_reception::{EventTransitionFromLabToReception, EventTransitionReceptionLaboratory};
    use crate::{TimePoint, TimeSpan};
    use crate::task_2::patient::PatientType;

    pub struct EventLaboratory {
        current_t: TimePoint,
        clinic: Rc<RefCell<Clinic>>,
        patient: Patient,
        assistant_index: usize
    }

    fn sample_delay() -> TimeSpan {
        TimeSpan(get_erlang_distribution(2, 4.0).sample())
    }

    pub enum EventLaboratoryTransitionResult {
        TransitionFromLabToReception(EventTransitionFromLabToReception),
        Terminal(EventTerminal)
    }

    impl EventLaboratory {
        pub fn get_current_t(&self) -> TimePoint {
            self.current_t
        }

        pub fn new(
            old_current_t: TimePoint,
            clinic: Rc<RefCell<Clinic>>,
            patient: Patient,
            assistant_index: usize
        ) -> Self {
            Self{current_t: old_current_t + sample_delay(), clinic, patient, assistant_index}
        }

        pub fn iterate(self) -> (Option<Self>, EventLaboratoryTransitionResult) {
            let transition_to = match self.patient.get_group() {
                PatientType::One => panic!("Patient one can not be in the laboratory"),
                PatientType::Two => EventLaboratoryTransitionResult::TransitionFromLabToReception(
                    EventTransitionFromLabToReception(
                        EventTransitionReceptionLaboratory::new(
                            self.current_t, self.clinic.clone(),
                            self.patient.upgrade_to_first_group()
                        )
                    )
                ),
                PatientType::Three => EventLaboratoryTransitionResult::Terminal(
                    EventTerminal::new(self.current_t)
                ),
            };
            let next_event = {
                let mut clinic = self.clinic.borrow_mut();
                if let Some(patient) = clinic.laboratory.queue.pop_front() {
                    Some(Self::new(self.current_t, self.clinic.clone(), patient, self.assistant_index))
                } else {
                    clinic.laboratory.is_lab_assistant_busy[self.assistant_index] = false;
                    None
                }
            };
            (next_event, transition_to)
        }
    }
}

enum Event {
    NewPatient(EventNewPatient),
    ReceptionDepartment(EventReceptionDepartment),
    FromReceptionLaboratory(EventTransitionFromReceptionToLaboratory),
    FromLabToReception(EventTransitionFromLabToReception),
    PatientWards(EventPatientWards),
    LabRegistration(EventLabRegistration),
    Laboratory(EventLaboratory),
    Terminal(EventTerminal),
}

impl Event {
    fn get_current_t(&self) -> TimePoint {
        match self {
            Event::NewPatient(event) => event.get_current_t(),
            Event::ReceptionDepartment(event) => event.get_current_t(),
            Event::FromReceptionLaboratory(event) => event.get_current_t(),
            Event::FromLabToReception(event) => event.get_current_t(),
            Event::PatientWards(event) => event.get_current_t(),
            Event::LabRegistration(event) => event.get_current_t(),
            Event::Laboratory(event) => event.get_current_t(),
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
    use std::cell::RefCell;
    use std::collections::BinaryHeap;
    use crate::task_2::{Event};
    use crate::task_2::event_laboratory::EventLaboratoryTransitionResult;
    use crate::task_2::event_reception_department::ReceptionDepartmentTransitionToResult;
    use crate::task_2::event_terminal::EventTerminal;
    use crate::TimePoint;

    #[test]
    fn test_general() {

        let mut nodes = BinaryHeap::new();

        nodes.push(Event::default());

        let end_time = TimePoint(10000.0);

        let last_event = loop {

            let next_event = nodes.pop().unwrap();
            if next_event.get_current_t() > end_time {
                break next_event;
            }
            match next_event {
                Event::NewPatient(event) => {
                    let res = event.iterate();
                    nodes.push(Event::NewPatient(res.0));
                    if let Some(next_event) = res.1 {
                        nodes.push(Event::ReceptionDepartment(next_event));
                    }
                },
                Event::ReceptionDepartment(event) => {
                    let res = event.iterate();
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
                    if let Some(res) = event.iterate() {
                        nodes.push(Event::LabRegistration(res));
                    }
                },
                Event::FromLabToReception(event) => {
                    if let Some(res) = event.iterate() {
                        nodes.push(Event::ReceptionDepartment(res));
                    }
                },
                Event::PatientWards(event) => {
                    let (self_event, _) = event.iterate();
                    if let Some(res_event) = self_event {
                        nodes.push(Event::PatientWards(res_event));
                    }
                },
                Event::LabRegistration(event) => {
                    let (self_event, next_event) = event.iterate();
                    if let Some(self_event) = self_event {
                        nodes.push(Event::LabRegistration(self_event));
                    }
                    if let Some(next_event) = next_event {
                        nodes.push(Event::Laboratory(next_event));
                    }
                },
                Event::Laboratory(event) => {
                    let (self_event, next_event) = event.iterate();
                    if let Some(self_event) = self_event {
                        nodes.push(Event::Laboratory(self_event));
                    }
                    match next_event {
                        EventLaboratoryTransitionResult::TransitionFromLabToReception(next_event) => {
                            nodes.push(Event::FromLabToReception(next_event));
                        }
                        EventLaboratoryTransitionResult::Terminal(_) => {}
                    }
                },
                Event::Terminal(event) => (),
            }
        };
    }

    #[test]
    fn test_event_ordering() {
        let mut bh = BinaryHeap::new();

        bh.push(Event::Terminal(EventTerminal::new(TimePoint(10.0))));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(9.0))));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(8.0))));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(14.0))));
        bh.push(Event::Terminal(EventTerminal::new(TimePoint(1.0))));

        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(1.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(8.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(9.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(10.0));
        assert_eq!(bh.pop().unwrap().get_current_t(), TimePoint(14.0));
    }

    #[derive(Debug, Clone)]
    struct Object {}
    #[test]
    fn test_ref_cell() {
        let obj = RefCell::new(Object{});
        let obj_mut_1 = obj.borrow_mut();
        drop(obj_mut_1);
        let obj_mut_2 = obj.borrow_mut();
        drop(obj_mut_2);
        let obj_mut_3 = obj.borrow_mut();
    }
}