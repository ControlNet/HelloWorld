
SET ECHO ON;

DROP SEQUENCE admission_admission_no_seq;

DROP SEQUENCE admission_procedure_admission_;

DROP SEQUENCE bed_bed_no_seq;

DROP SEQUENCE bed_type_bed_type_code_seq;

DROP SEQUENCE cost_center_cost_centre_code;

DROP SEQUENCE doctor_doc_id_seq;

DROP SEQUENCE item_item_code_seq;

DROP SEQUENCE nurse_nurse_id_seq;

DROP SEQUENCE patient_patient_id_seq;

DROP SEQUENCE procedure_procedure_code_seq;

DROP SEQUENCE specilisation_spec_no_seq;

DROP SEQUENCE ward_ward_code_seq;

DROP TABLE admission CASCADE CONSTRAINTS;

DROP TABLE admission_bed CASCADE CONSTRAINTS;

DROP TABLE admission_procedure CASCADE CONSTRAINTS;

DROP TABLE bed CASCADE CONSTRAINTS;

DROP TABLE bed_type CASCADE CONSTRAINTS;

DROP TABLE cost_center CASCADE CONSTRAINTS;

DROP TABLE doctor CASCADE CONSTRAINTS;

DROP TABLE doctor_specilisation CASCADE CONSTRAINTS;

DROP TABLE item CASCADE CONSTRAINTS;

DROP TABLE nurse CASCADE CONSTRAINTS;

DROP TABLE nurse_assignment CASCADE CONSTRAINTS;

DROP TABLE patient CASCADE CONSTRAINTS;

DROP TABLE procedure CASCADE CONSTRAINTS;

DROP TABLE procedure_item CASCADE CONSTRAINTS;

DROP TABLE specilisation CASCADE CONSTRAINTS;

DROP TABLE ward CASCADE CONSTRAINTS;

CREATE TABLE admission (
    admission_no          NUMBER(7) NOT NULL,
    patient_id            NUMBER(7) NOT NULL,
    admission_datetime    DATE NOT NULL,
    discharged_datetime   DATE NOT NULL,
    doc_id                NUMBER(7) NOT NULL
);

COMMENT ON COLUMN admission.admission_no IS
    'admission_no: admission number';

COMMENT ON COLUMN admission.patient_id IS
    'patient_id: patient ID';

COMMENT ON COLUMN admission.admission_datetime IS
    'admission_datetime: admission date and time';

COMMENT ON COLUMN admission.discharged_datetime IS
    'discharged_datetime: discharged date and time';

COMMENT ON COLUMN admission.doc_id IS
    'doc_id: doctor ID';

ALTER TABLE admission ADD CONSTRAINT admission_pk PRIMARY KEY ( admission_no );

ALTER TABLE admission ADD CONSTRAINT admission_unique UNIQUE ( patient_id,
                                                               admission_datetime
                                                               );

CREATE TABLE admission_bed (
    admission_no            NUMBER(7) NOT NULL,
    bed_assigned_datetime   DATE NOT NULL,
    ward_code               NUMBER(7) NOT NULL,
    bed_no                  NUMBER(3) NOT NULL
);

COMMENT ON COLUMN admission_bed.admission_no IS
    'admission_no: admission number';

COMMENT ON COLUMN admission_bed.bed_assigned_datetime IS
    'bed_assigned_datetime: bed assigned datetime';

COMMENT ON COLUMN admission_bed.ward_code IS
    'ward_code:ward code';

COMMENT ON COLUMN admission_bed.bed_no IS
    'bed_no: bed number';

ALTER TABLE admission_bed ADD CONSTRAINT admission_bed_pk PRIMARY KEY ( admission_no
,
                                                                        bed_assigned_datetime
                                                                        );

CREATE TABLE admission_procedure (
    admission_procedure_no   NUMBER(7) NOT NULL,
    admission_no             NUMBER(7) NOT NULL,
    procedure_datetime       DATE NOT NULL,
    procedure_charge         NUMBER(7, 2) NOT NULL,
    total_extra_charge       NUMBER(7, 2) NOT NULL,
    requested_doc_id         NUMBER(7) NOT NULL,
    carried_doc_id           NUMBER(7),
    procedure_code           NUMBER(7) NOT NULL
);

COMMENT ON COLUMN admission_procedure.admission_procedure_no IS
    'admission_procedure_no: admission procedure number';

COMMENT ON COLUMN admission_procedure.admission_no IS
    'admission_no: admission number';

COMMENT ON COLUMN admission_procedure.procedure_datetime IS
    'procedure_datetime: procedure date and time';

COMMENT ON COLUMN admission_procedure.procedure_charge IS
    'procedure_charge: procedure charge';

COMMENT ON COLUMN admission_procedure.total_extra_charge IS
    'total_extra_charge: total extra charge';

COMMENT ON COLUMN admission_procedure.requested_doc_id IS
    'doc_id: doctor ID';

COMMENT ON COLUMN admission_procedure.carried_doc_id IS
    'doc_id: doctor ID';

COMMENT ON COLUMN admission_procedure.procedure_code IS
    'procedure_code: procedure code';

ALTER TABLE admission_procedure ADD CONSTRAINT admission_procedure_pk PRIMARY KEY
( admission_procedure_no );

ALTER TABLE admission_procedure ADD CONSTRAINT admission_procedure_unique UNIQUE

( procedure_datetime,
                                                                                   admission_no
                                                                                   )
                                                                                   ;

CREATE TABLE bed (
    ward_code       NUMBER(7) NOT NULL,
    bed_no          NUMBER(3) NOT NULL,
    bed_phone       VARCHAR2(10),
    bed_type_code   NUMBER(7) NOT NULL
);

COMMENT ON COLUMN bed.ward_code IS
    'ward_code:ward code';

COMMENT ON COLUMN bed.bed_no IS
    'bed_no: bed number';

COMMENT ON COLUMN bed.bed_phone IS
    'bed_phone: bedside phone number';

COMMENT ON COLUMN bed.bed_type_code IS
    'bed_type_code: bed type code';

ALTER TABLE bed ADD CONSTRAINT bed_pk PRIMARY KEY ( ward_code,
                                                    bed_no );

CREATE TABLE bed_type (
    bed_type_code   NUMBER(7) NOT NULL,
    bed_type_desc   VARCHAR2(50) NOT NULL
);

COMMENT ON COLUMN bed_type.bed_type_code IS
    'bed_type_code: bed type code';

COMMENT ON COLUMN bed_type.bed_type_desc IS
    'bed_type_desc: bed type description';

ALTER TABLE bed_type ADD CONSTRAINT bed_type_pk PRIMARY KEY ( bed_type_code );

ALTER TABLE bed_type ADD CONSTRAINT bed_type_unique UNIQUE ( bed_type_desc );

CREATE TABLE cost_center (
    cost_centre_code    NUMBER(7) NOT NULL,
    cost_centre_title   VARCHAR2(50) NOT NULL,
    manager_fname       VARCHAR2(50) NOT NULL,
    manager_lname       VARCHAR2(50) NOT NULL
);

COMMENT ON COLUMN cost_center.cost_centre_code IS
    'cost_centre_code: cost centre code';

COMMENT ON COLUMN cost_center.cost_centre_title IS
    'cost_centre_title: cost centre title';

COMMENT ON COLUMN cost_center.manager_fname IS
    'manager_fname: manager first name';

COMMENT ON COLUMN cost_center.manager_lname IS
    'manager_lname: manager last name';

ALTER TABLE cost_center ADD CONSTRAINT cost_center_pk PRIMARY KEY ( cost_centre_code
);

CREATE TABLE doctor (
    doc_id      NUMBER(7) NOT NULL,
    doc_title   VARCHAR2(50) NOT NULL,
    doc_fname   VARCHAR2(50) NOT NULL,
    doc_lname   VARCHAR2(50) NOT NULL,
    doc_phone   CHAR(10) NOT NULL
);

COMMENT ON COLUMN doctor.doc_id IS
    'doc_id: doctor ID';

COMMENT ON COLUMN doctor.doc_title IS
    'doc_title: doctor title';

COMMENT ON COLUMN doctor.doc_fname IS
    'doc_fname; doctor first name';

COMMENT ON COLUMN doctor.doc_lname IS
    'doc_lname: doctor last name';

COMMENT ON COLUMN doctor.doc_phone IS
    'doc_phone: doctor phone number';

ALTER TABLE doctor ADD CONSTRAINT doctor_pk PRIMARY KEY ( doc_id );

CREATE TABLE doctor_specilisation (
    doc_id    NUMBER(7) NOT NULL,
    spec_no   NUMBER(7) NOT NULL
);

COMMENT ON COLUMN doctor_specilisation.doc_id IS
    'doc_id: doctor ID';

COMMENT ON COLUMN doctor_specilisation.spec_no IS
    'spec_no: specilisation number';

ALTER TABLE doctor_specilisation ADD CONSTRAINT doctor_specilisation_pk PRIMARY KEY
( doc_id,
                                                                                      spec_no
                                                                                      )
                                                                                      ;

CREATE TABLE item (
    item_code          NUMBER(7) NOT NULL,
    item_desc          VARCHAR2(50) NOT NULL,
    item_stock         NUMBER(10) NOT NULL,
    item_price         NUMBER(7, 2) NOT NULL,
    cost_centre_code   NUMBER(7) NOT NULL
);

COMMENT ON COLUMN item.item_code IS
    'item_code: item code';

COMMENT ON COLUMN item.item_desc IS
    'item_desc: item description';

COMMENT ON COLUMN item.item_stock IS
    'item_stock: item amount in stock';

COMMENT ON COLUMN item.item_price IS
    'item_price: item price';

COMMENT ON COLUMN item.cost_centre_code IS
    'cost_centre_code: cost centre code';

ALTER TABLE item ADD CONSTRAINT item_pk PRIMARY KEY ( item_code );

CREATE TABLE nurse (
    nurse_id              NUMBER(7) NOT NULL,
    nurse_fname           VARCHAR2(50) NOT NULL,
    nurse_lname           VARCHAR2(50) NOT NULL,
    nurse_phone           CHAR(10) NOT NULL,
    nurse_with_children   CHAR(1) NOT NULL
);

ALTER TABLE nurse
    ADD CONSTRAINT nurse_check CHECK ( nurse_with_children IN (
        'N',
        'Y'
    ) );

COMMENT ON COLUMN nurse.nurse_id IS
    'nurse_id: nurse ID';

COMMENT ON COLUMN nurse.nurse_fname IS
    'nurse_fname: nurse first name';

COMMENT ON COLUMN nurse.nurse_lname IS
    'nurse_lname: nurse last name';

COMMENT ON COLUMN nurse.nurse_phone IS
    'nurse_phone: nurse phone number';

COMMENT ON COLUMN nurse.nurse_with_children IS
    'nurse_with_children: cetrified work with children';

ALTER TABLE nurse ADD CONSTRAINT nurse_pk PRIMARY KEY ( nurse_id );

CREATE TABLE nurse_assignment (
    nurse_id         NUMBER(7) NOT NULL,
    date_assgined    DATE NOT NULL,
    date_completed   DATE NOT NULL,
    ward_code        NUMBER(7) NOT NULL
);

COMMENT ON COLUMN nurse_assignment.nurse_id IS
    'nurse_id: nurse ID';

COMMENT ON COLUMN nurse_assignment.date_assgined IS
    'date_assigned: date assgined';

COMMENT ON COLUMN nurse_assignment.date_completed IS
    'date_completed: date completed';

COMMENT ON COLUMN nurse_assignment.ward_code IS
    'ward_code:ward code';

ALTER TABLE nurse_assignment ADD CONSTRAINT nurse_assignment_pk PRIMARY KEY ( nurse_id
,
                                                                              date_assgined
                                                                              );

CREATE TABLE patient (
    patient_id               NUMBER(7) NOT NULL,
    patient_fname            VARCHAR2(50) NOT NULL,
    patient_lname            VARCHAR2(50) NOT NULL,
    patient_add              VARCHAR2(50) NOT NULL,
    patient_dob              DATE NOT NULL,
    patient_contact_number   CHAR(10) NOT NULL
);

COMMENT ON COLUMN patient.patient_id IS
    'patient_id: patient ID';

COMMENT ON COLUMN patient.patient_fname IS
    'patient_fname; patient first name';

COMMENT ON COLUMN patient.patient_lname IS
    'patient_lname: patient last name';

COMMENT ON COLUMN patient.patient_add IS
    'patient_add: patient address';

COMMENT ON COLUMN patient.patient_dob IS
    'patient_dob; patient date of birth';

COMMENT ON COLUMN patient.patient_contact_number IS
    'patient_contact_number: patient emergency contact number';

ALTER TABLE patient ADD CONSTRAINT patient_pk PRIMARY KEY ( patient_id );

CREATE TABLE procedure (
    procedure_code            NUMBER(7) NOT NULL,
    procedure_name            VARCHAR2(50) NOT NULL,
    procedure_desc            VARCHAR2(50) NOT NULL,
    procedure_time            VARCHAR2(50) NOT NULL,
    procedure_standard_cost   NUMBER(7, 2) NOT NULL
);

COMMENT ON COLUMN procedure.procedure_code IS
    'procedure_code: procedure code';

COMMENT ON COLUMN procedure.procedure_name IS
    'procedure_name; procedure name';

COMMENT ON COLUMN procedure.procedure_desc IS
    'procedure_desc: procedure description';

COMMENT ON COLUMN procedure.procedure_time IS
    'procedure_time: time used of the procedure';

COMMENT ON COLUMN procedure.procedure_standard_cost IS
    'procedure_standard_cost: procedure standard cost';

ALTER TABLE procedure ADD CONSTRAINT procedure_pk PRIMARY KEY ( procedure_code );

CREATE TABLE procedure_item (
    admission_procedure_no   NUMBER(7) NOT NULL,
    item_code                NUMBER(7) NOT NULL,
    item_qty                 NUMBER(10) NOT NULL,
    total_item_charge        NUMBER(7, 2) NOT NULL,
    admission_no             NUMBER(7) NOT NULL
);

COMMENT ON COLUMN procedure_item.admission_procedure_no IS
    'admission_procedure_no: admission procedure number';

COMMENT ON COLUMN procedure_item.item_code IS
    'item_code: item code';

COMMENT ON COLUMN procedure_item.item_qty IS
    'item_qty: item quantity';

COMMENT ON COLUMN procedure_item.total_item_charge IS
    'total_item_charge: total item charge';

ALTER TABLE procedure_item ADD CONSTRAINT procedure_item_pk PRIMARY KEY ( admission_procedure_no
,
                                                                          item_code
                                                                          );

CREATE TABLE specilisation (
    spec_no     NUMBER(7) NOT NULL,
    spec_desc   VARCHAR2(50) NOT NULL
);

COMMENT ON COLUMN specilisation.spec_no IS
    'spec_no: specilisation number';

COMMENT ON COLUMN specilisation.spec_desc IS
    'spec_desc: specilisation description';

ALTER TABLE specilisation ADD CONSTRAINT specilisation_pk PRIMARY KEY ( spec_no )
;

ALTER TABLE specilisation ADD CONSTRAINT specilisation_unique UNIQUE ( spec_desc

);

CREATE TABLE ward (
    ward_code              NUMBER(7) NOT NULL,
    total_bed_amount       NUMBER(3) NOT NULL,
    available_bed_amount   NUMBER(3) NOT NULL
);

COMMENT ON COLUMN ward.ward_code IS
    'ward_code:ward code';

COMMENT ON COLUMN ward.total_bed_amount IS
    'total_bed_amount: total bed amount';

COMMENT ON COLUMN ward.available_bed_amount IS
    'available_bed_amount: available bed amount';

ALTER TABLE ward ADD CONSTRAINT ward_pk PRIMARY KEY ( ward_code );

ALTER TABLE procedure_item
    ADD CONSTRAINT adm_pro_pro_item FOREIGN KEY ( admission_procedure_no )
        REFERENCES admission_procedure ( admission_procedure_no );

ALTER TABLE admission_bed
    ADD CONSTRAINT admission_admission_bed FOREIGN KEY ( admission_no )
        REFERENCES admission ( admission_no );

ALTER TABLE admission_procedure
    ADD CONSTRAINT admission_admission_procedure FOREIGN KEY ( admission_no )
        REFERENCES admission ( admission_no );

ALTER TABLE admission_bed
    ADD CONSTRAINT bed_admission_bed FOREIGN KEY ( ward_code,
                                                   bed_no )
        REFERENCES bed ( ward_code,
                         bed_no );

ALTER TABLE bed
    ADD CONSTRAINT bed_type_bed FOREIGN KEY ( bed_type_code )
        REFERENCES bed_type ( bed_type_code );

ALTER TABLE item
    ADD CONSTRAINT cost_centre_item FOREIGN KEY ( cost_centre_code )
        REFERENCES cost_center ( cost_centre_code );

ALTER TABLE admission_procedure
    ADD CONSTRAINT doc_adm_pro_carries FOREIGN KEY ( carried_doc_id )
        REFERENCES doctor ( doc_id );

ALTER TABLE admission_procedure
    ADD CONSTRAINT doc_adm_pro_requests FOREIGN KEY ( requested_doc_id )
        REFERENCES doctor ( doc_id );

ALTER TABLE admission
    ADD CONSTRAINT doctor_admission FOREIGN KEY ( doc_id )
        REFERENCES doctor ( doc_id );

ALTER TABLE doctor_specilisation
    ADD CONSTRAINT doctor_doctor_specilisation FOREIGN KEY ( doc_id )
        REFERENCES doctor ( doc_id );

ALTER TABLE procedure_item
    ADD CONSTRAINT item_procedure_item FOREIGN KEY ( item_code )
        REFERENCES item ( item_code );

ALTER TABLE nurse_assignment
    ADD CONSTRAINT nurse_nurse_assignment FOREIGN KEY ( nurse_id )
        REFERENCES nurse ( nurse_id );

ALTER TABLE admission
    ADD CONSTRAINT patient_admission FOREIGN KEY ( patient_id )
        REFERENCES patient ( patient_id );

ALTER TABLE admission_procedure
    ADD CONSTRAINT procedure_admission_procedure FOREIGN KEY ( procedure_code )
        REFERENCES procedure ( procedure_code );

ALTER TABLE doctor_specilisation
    ADD CONSTRAINT spec_doctor_spec FOREIGN KEY ( spec_no )
        REFERENCES specilisation ( spec_no );

ALTER TABLE bed
    ADD CONSTRAINT ward_bed FOREIGN KEY ( ward_code )
        REFERENCES ward ( ward_code );

ALTER TABLE nurse_assignment
    ADD CONSTRAINT ward_nurse_assignment FOREIGN KEY ( ward_code )
        REFERENCES ward ( ward_code );

CREATE SEQUENCE admission_admission_no_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE admission_procedure_admission_ START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE bed_bed_no_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE bed_type_bed_type_code_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE cost_center_cost_centre_code START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE doctor_doc_id_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE item_item_code_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE nurse_nurse_id_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE patient_patient_id_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE procedure_procedure_code_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE specilisation_spec_no_seq START WITH 1 NOCACHE ORDER;

CREATE SEQUENCE ward_ward_code_seq START WITH 1 NOCACHE ORDER;



-- Oracle SQL Developer Data Modeler Summary Report: 
-- 
-- CREATE TABLE                            16
-- CREATE INDEX                             0
-- ALTER TABLE                             38
-- CREATE VIEW                              0
-- ALTER VIEW                               0
-- CREATE PACKAGE                           0
-- CREATE PACKAGE BODY                      0
-- CREATE PROCEDURE                         0
-- CREATE FUNCTION                          0
-- CREATE TRIGGER                           0
-- ALTER TRIGGER                            0
-- CREATE COLLECTION TYPE                   0
-- CREATE STRUCTURED TYPE                   0
-- CREATE STRUCTURED TYPE BODY              0
-- CREATE CLUSTER                           0
-- CREATE CONTEXT                           0
-- CREATE DATABASE                          0
-- CREATE DIMENSION                         0
-- CREATE DIRECTORY                         0
-- CREATE DISK GROUP                        0
-- CREATE ROLE                              0
-- CREATE ROLLBACK SEGMENT                  0
-- CREATE SEQUENCE                         12
-- CREATE MATERIALIZED VIEW                 0
-- CREATE MATERIALIZED VIEW LOG             0
-- CREATE SYNONYM                           0
-- CREATE TABLESPACE                        0
-- CREATE USER                              0
-- 
-- DROP TABLESPACE                          0
-- DROP DATABASE                            0
-- 
-- REDACTION POLICY                         0
-- 
-- ORDS DROP SCHEMA                         0
-- ORDS ENABLE SCHEMA                       0
-- ORDS ENABLE OBJECT                       0
-- 
-- ERRORS                                   0
-- WARNINGS                                 0

SET ECHO OFF;