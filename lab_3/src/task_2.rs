use std::collections::BinaryHeap;
use rand::distributions::{Distribution, Uniform};
use rand_distr::Exp;
use crate::utils::TimePoint;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
enum Patient {
    Three,
    Two,
    One,
}

struct Clinic {
    reception_department: ReceptionDepartment
}

struct Doctor {
    is_acquired: bool
}

struct ReceptionDepartment {
    queue: BinaryHeap<Patient>,
    is_doctor_busy: [bool; 2],
}

mod create_patient {
    use std::cell::RefCell;
    use std::rc::Rc;
    use rand::distributions::{Distribution, Uniform};
    use rand_distr::Exp;
    use crate::task_2::event_reception_department::EventReceptionDepartment;
    use crate::task_2::{Clinic, Patient};
    use crate::TimeSpan;
    use crate::utils::TimePoint;

    struct EventNewPatient {
        current_t: TimePoint,
        patient: Patient,
        clinic: Rc<RefCell<Clinic>>
    }

    static DELAY_GEN: Exp<f64> = Exp::new(15.0).expect("Failed to create delay gen");

    fn generate_patient() -> Patient {
        let value = Uniform::new(0.0, 1.0).sample(&mut rand::thread_rng());
        match value {
            ..0.5 => Patient::One,
            0.5..0.6 => Patient::Two,
            0.6.. => Patient::Three,
            _ => panic!("PatientType::generate_patient error")
        }
    }

    impl EventNewPatient {
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
            (
                Self {
                    current_t: self.current_t + TimeSpan(DELAY_GEN.sample(&mut rand::thread_rng())),
                    patient: generate_patient(),
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
    use crate::TimeSpan;
    use crate::utils::TimePoint;

    pub struct EventReceptionDepartment {
        current_t: TimePoint,
        doctor_index: usize,
        clinic: Rc<RefCell<Clinic>>,
        patient: Patient
    }

    fn determine_delay(patient: Patient) -> TimeSpan {
        match patient {
            Patient::One => TimeSpan(15.0),
            Patient::Two => TimeSpan(40.0),
            Patient::Three => TimeSpan(30.0),
        }
    }

    enum TransitionToResult {
        ToPatientWard(),
        ToLaboratory()
    }

    impl EventReceptionDepartment {
        pub fn new(old_current_t: TimePoint, doctor_index: usize, clinic: Rc<RefCell<Clinic>>, patient: Patient) -> Self {
            Self{current_t: old_current_t + determine_delay(patient), doctor_index, clinic, patient}
        }

        pub fn iterate(self) -> (Option<EventReceptionDepartment>, TransitionToResult) {
            let transition_to = match self.patient {
                Patient::One => TransitionToResult::ToPatientWard(),
                Patient::Two | Patient::Three => TransitionToResult::ToLaboratory(),
            };
            let next_reception_dep = {
                let mut clinic = self.clinic.borrow_mut();
                if let Some(patient) = clinic.reception_department.queue.pop() {
                    Some(Self::new(self.current_t, self.doctor_index, self.clinic.clone(), patient))
                } else {
                    None
                }
            };
            (next_reception_dep, transition_to)
        }
    }

}

