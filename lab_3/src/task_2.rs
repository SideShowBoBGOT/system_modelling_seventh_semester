use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::rc::Rc;
use rand::distributions::{Distribution, Uniform};
use rand::{thread_rng, RngCore};
use rand_distr::Exp;
use crate::QueueSize;
use crate::utils::TimePoint;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
enum Patient {
    Three,
    Two,
    One,
}

struct Clinic {
    reception_department: ReceptionDepartment,
    patient_wards: PatientWards
}

struct ReceptionDepartment {
    queue: BinaryHeap<Patient>,
    is_doctor_busy: [bool; 2],
}

struct PatientWards {
    queue_size: usize,
    is_attendant_busy: [bool; 3],
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
    use crate::task_2::event_patient_wards::EventPatientWards;
    use crate::task_2::event_transition_from_reception_to_lab::EventTransitionFromReceptionToLaboratory;
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
        EventPatientWards(Option<EventPatientWards>),
        EventTransitionFromReceptionToLaboratory(EventTransitionFromReceptionToLaboratory)
    }

    impl EventReceptionDepartment {
        pub fn new(old_current_t: TimePoint, doctor_index: usize, clinic: Rc<RefCell<Clinic>>, patient: Patient) -> Self {
            Self{current_t: old_current_t + determine_delay(patient), doctor_index, clinic, patient}
        }

        pub fn iterate(self) -> (Option<EventReceptionDepartment>, TransitionToResult) {
            let transition_to = match self.patient {
                Patient::One => TransitionToResult::EventPatientWards (
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
                Patient::Two | Patient::Three => {
                    TransitionToResult::EventTransitionFromReceptionToLaboratory({
                        EventTransitionFromReceptionToLaboratory::new(
                            self.current_t, self.clinic.clone(), self.patient
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

static RECEPTION_LABORATORY_TRANSITION_DELAY: Uniform<f64> = Uniform::new(2.0, 5.0);

mod event_transition_from_reception_to_lab {
    use std::cell::RefCell;
    use std::rc::Rc;
    use rand::distributions::Distribution;
    use crate::task_2::{Clinic, Patient, RECEPTION_LABORATORY_TRANSITION_DELAY};
    use crate::task_2::event_registration::EventRegistration;
    use crate::TimeSpan;
    use crate::utils::TimePoint;

    pub struct EventTransitionFromReceptionToLaboratory {
        current_t: TimePoint,
        clinic: Rc<RefCell<Clinic>>,
        patient: Patient,
    }

    impl EventTransitionFromReceptionToLaboratory {
        pub fn new(old_current_t: TimePoint, clinic: Rc<RefCell<Clinic>>, patient: Patient) -> Self {
            let delay = TimeSpan(RECEPTION_LABORATORY_TRANSITION_DELAY.sample(&mut rand::thread_rng()));
            Self{current_t: old_current_t + delay, clinic, patient}
        }

        pub fn iterate(self) -> EventRegistration {

        }
    }
}

mod event_patient_wards {
    use std::cell::RefCell;
    use std::rc::Rc;
    use rand::distributions::{Distribution, Uniform};
    use crate::task_2::Clinic;
    use crate::TimeSpan;
    use crate::utils::TimePoint;

    pub struct EventPatientWards {
        current_t: TimePoint,
        clinic: Rc<RefCell<Clinic>>,
        attendant_index: usize
    }

    static DELAY_GEN: Uniform<f64> = Uniform::new(3.0, 8.0);

    impl EventPatientWards {
        pub fn new(old_current_t: TimePoint, clinic: Rc<RefCell<Clinic>>, attendant_index: usize) -> Self {
            let delay = TimeSpan(DELAY_GEN.sample(&mut rand::thread_rng()));
            Self{current_t: old_current_t + delay, clinic, attendant_index}
        }

        pub fn iterate(self) -> Option<Self> {
            let mut clinic = self.clinic.borrow_mut();
            if clinic.patient_wards.queue_size > 0 {
                clinic.patient_wards.queue_size -= 1;
                Some(Self::new(self.current_t, self.clinic, self.attendant_index))
            } else {
                clinic.patient_wards.is_attendant_busy[self.attendant_index] = false;
                None
            }
        }
    }
}

mod erlang_distribution {
    use std::cell::RefCell;
    use rand::{thread_rng, RngCore};

    pub struct ErlangDistribution(RefCell<rand_simple::Erlang>);

    impl ErlangDistribution {
        pub fn new(shape: i64, scale: f64) -> Self {
            let mut erlang = rand_simple::Erlang::new(
                [
                    thread_rng().next_u32(),
                    thread_rng().next_u32(),
                    thread_rng().next_u32(),
                ]
            );
            erlang.try_set_params(shape, scale).expect("Erlang set params failed");
            Self(RefCell::new(erlang))
        }

        pub fn sample(&self) -> f64 {
            self.0.borrow_mut().sample()
        }
    }
}

mod event_registration {
    use std::cell::RefCell;
    use std::rc::Rc;
    use crate::task_2::{Clinic, Patient};
    use crate::task_2::erlang_distribution::ErlangDistribution;
    use crate::TimeSpan;
    use crate::utils::TimePoint;

    static DELAY_GEN: ErlangDistribution = ErlangDistribution::new(3, 4.5);

    pub struct EventRegistration {
        current_t: TimePoint,
        clinic: Rc<RefCell<Clinic>>,
        patient: Patient
    }

    impl EventRegistration {
        pub fn new(
            old_current_t: TimePoint,
            clinic: Rc<RefCell<Clinic>>,
            patient: Patient
        ) -> Self {
            let delay = TimeSpan(DELAY_GEN.sample());
            Self{current_t: old_current_t + delay, clinic, attendant_index}
        }
    }
}


