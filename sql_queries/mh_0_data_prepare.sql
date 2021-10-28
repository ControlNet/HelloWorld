-- Provided in material.

DROP TABLE adm_prc CASCADE CONSTRAINTS;

DROP TABLE admission CASCADE CONSTRAINTS;

DROP TABLE costcentre CASCADE CONSTRAINTS;

DROP TABLE doctor CASCADE CONSTRAINTS;

DROP TABLE doctor_speciality CASCADE CONSTRAINTS;

DROP TABLE item CASCADE CONSTRAINTS;

DROP TABLE item_treatment CASCADE CONSTRAINTS;

DROP TABLE patient CASCADE CONSTRAINTS;

DROP TABLE procedure CASCADE CONSTRAINTS;

DROP TABLE speciality CASCADE CONSTRAINTS;

CREATE TABLE adm_prc (
    adprc_no           NUMBER(7) NOT NULL,
    adprc_date_time    DATE NOT NULL,
    adprc_pat_cost     NUMBER(7, 2) NOT NULL,
    adprc_items_cost   NUMBER(6, 2) NOT NULL,
    adm_no             NUMBER(6) NOT NULL,
    proc_code          NUMBER(5) NOT NULL,
    request_dr_id      NUMBER(4) NOT NULL,
    perform_dr_id      NUMBER(4)
);

COMMENT ON COLUMN adm_prc.adprc_no IS
    'Admission procedure identifier (PK)';

COMMENT ON COLUMN adm_prc.adprc_date_time IS
    'Date and time this procedure was carried out for this admission';

COMMENT ON COLUMN adm_prc.adprc_pat_cost IS
    'Charge to patient for this procedure';

COMMENT ON COLUMN adm_prc.adprc_items_cost IS
    'Total patient charge for extra items required';

COMMENT ON COLUMN adm_prc.adm_no IS
    'Admission number (PK)';

COMMENT ON COLUMN adm_prc.proc_code IS
    'Procedure code (PK)';

COMMENT ON COLUMN adm_prc.request_dr_id IS
    'Doctor id (PK)';

COMMENT ON COLUMN adm_prc.perform_dr_id IS
    'Doctor id (PK)';

ALTER TABLE adm_prc ADD CONSTRAINT adm_prc_pk PRIMARY KEY ( adprc_no );

ALTER TABLE adm_prc ADD CONSTRAINT adm_prc_nk UNIQUE ( adprc_date_time,
                                                       adm_no );

CREATE TABLE admission (
    adm_no          NUMBER(6) NOT NULL,
    adm_date_time   DATE NOT NULL,
    adm_discharge   DATE,
    patient_id      NUMBER(6) NOT NULL,
    doctor_id       NUMBER(4) NOT NULL
);

COMMENT ON COLUMN admission.adm_no IS
    'Admission number (PK)';

COMMENT ON COLUMN admission.adm_date_time IS
    'Admisison date and time';

COMMENT ON COLUMN admission.patient_id IS
    'Patient identifier (PK)';

COMMENT ON COLUMN admission.doctor_id IS
    'Doctor id (PK)';

ALTER TABLE admission ADD CONSTRAINT admission_pk PRIMARY KEY ( adm_no );

ALTER TABLE admission ADD CONSTRAINT admission_nk UNIQUE ( patient_id,
                                                           adm_date_time );

CREATE TABLE costcentre (
    cc_code           CHAR(5) NOT NULL,
    cc_title          VARCHAR2(50) NOT NULL,
    cc_manager_name   VARCHAR2(80) NOT NULL
);

COMMENT ON COLUMN costcentre.cc_code IS
    'Cost Centre Code (PK)';

COMMENT ON COLUMN costcentre.cc_title IS
    'Name of Cost Centre';

COMMENT ON COLUMN costcentre.cc_manager_name IS
    'Name of Cost Centre Manager';

ALTER TABLE costcentre ADD CONSTRAINT costcentre_pk PRIMARY KEY ( cc_code );

CREATE TABLE doctor (
    doctor_id      NUMBER(4) NOT NULL,
    doctor_title   VARCHAR2(2) NOT NULL,
    doctor_fname   VARCHAR2(50),
    doctor_lname   VARCHAR2(50),
    doctor_phone   CHAR(10) NOT NULL
);

COMMENT ON COLUMN doctor.doctor_id IS
    'Doctor id (PK)';

COMMENT ON COLUMN doctor.doctor_title IS
    'MR, MS, DR';

COMMENT ON COLUMN doctor.doctor_fname IS
    'Doctor first name';

COMMENT ON COLUMN doctor.doctor_lname IS
    'Doctor Last Name';

COMMENT ON COLUMN doctor.doctor_phone IS
    'Doctor contact number';

ALTER TABLE doctor ADD CONSTRAINT doctor_pk PRIMARY KEY ( doctor_id );

CREATE TABLE doctor_speciality (
    spec_code   CHAR(6) NOT NULL,
    doctor_id   NUMBER(4) NOT NULL
);

COMMENT ON COLUMN doctor_speciality.spec_code IS
    'Speciality code for doctor (PK)';

COMMENT ON COLUMN doctor_speciality.doctor_id IS
    'Doctor id (PK)';

ALTER TABLE doctor_speciality ADD CONSTRAINT doctor_speciality_pk PRIMARY KEY ( spec_code
,
                                                                                doctor_id
                                                                                )
                                                                                ;

CREATE TABLE item (
    item_code          CHAR(5) NOT NULL,
    item_description   VARCHAR2(100) NOT NULL,
    item_stock         NUMBER(3) NOT NULL,
    item_cost          NUMBER(7, 2) NOT NULL,
    cc_code            CHAR(5) NOT NULL
);

COMMENT ON COLUMN item.item_code IS
    'Item Code (PK)';

COMMENT ON COLUMN item.item_description IS
    'Description of item';

COMMENT ON COLUMN item.item_stock IS
    'Current stock of item';

COMMENT ON COLUMN item.item_cost IS
    'Item cost for this item';

COMMENT ON COLUMN item.cc_code IS
    'Cost Centre Code (PK)';

ALTER TABLE item ADD CONSTRAINT item_pk PRIMARY KEY ( item_code );

CREATE TABLE item_treatment (
    adprc_no             NUMBER(7) NOT NULL,
    item_code            CHAR(5) NOT NULL,
    it_qty_used          NUMBER(2) NOT NULL,
    it_item_total_cost   NUMBER(8, 2) NOT NULL
);

COMMENT ON COLUMN item_treatment.adprc_no IS
    'Admission procedure identifier (PK)';

COMMENT ON COLUMN item_treatment.item_code IS
    'Item Code (PK)';

COMMENT ON COLUMN item_treatment.it_qty_used IS
    'Quantity of item used in this admission procedure';

COMMENT ON COLUMN item_treatment.it_item_total_cost IS
    'Total items cost for items used in this admission procedure';

ALTER TABLE item_treatment ADD CONSTRAINT item_treatment_pk PRIMARY KEY ( adprc_no
,
                                                                          item_code
                                                                          );

CREATE TABLE patient (
    patient_id            NUMBER(6) NOT NULL,
    patient_fname         VARCHAR2(50),
    patient_lname         VARCHAR2(50),
    patient_address       VARCHAR2(100) NOT NULL,
    patient_dob           DATE NOT NULL,
    patient_contact_phn   CHAR(10) NOT NULL
);

COMMENT ON COLUMN patient.patient_id IS
    'Patient identifier (PK)';

COMMENT ON COLUMN patient.patient_fname IS
    'Patient first name';

COMMENT ON COLUMN patient.patient_lname IS
    'Patient last name';

COMMENT ON COLUMN patient.patient_address IS
    'Patient Adress';

COMMENT ON COLUMN patient.patient_dob IS
    'Patient data of Birth';

COMMENT ON COLUMN patient.patient_contact_phn IS
    'Patient contact phone number';

ALTER TABLE patient ADD CONSTRAINT patient_pk PRIMARY KEY ( patient_id );

CREATE TABLE procedure (
    proc_code          NUMBER(5) NOT NULL,
    proc_name          VARCHAR2(100) NOT NULL,
    proc_description   VARCHAR2(300) NOT NULL,
    proc_time          NUMBER(3) NOT NULL,
    proc_std_cost      NUMBER(7, 2) NOT NULL
);

COMMENT ON COLUMN procedure.proc_code IS
    'Procedure code (PK)';

COMMENT ON COLUMN procedure.proc_name IS
    'Procedure Name';

COMMENT ON COLUMN procedure.proc_description IS
    'Procedure Description';

COMMENT ON COLUMN procedure.proc_time IS
    'Standard time required for this procedure in mins';

COMMENT ON COLUMN procedure.proc_std_cost IS
    'Standard patient charge for procedure';

ALTER TABLE procedure ADD CONSTRAINT procedure_pk PRIMARY KEY ( proc_code );

ALTER TABLE procedure ADD CONSTRAINT proc_name_unq UNIQUE ( proc_name );

CREATE TABLE speciality (
    spec_code          CHAR(6) NOT NULL,
    spec_description   VARCHAR2(50) NOT NULL
);

COMMENT ON COLUMN speciality.spec_code IS
    'Speciality code for doctor (PK)';

COMMENT ON COLUMN speciality.spec_description IS
    'Description of speciality code';

ALTER TABLE speciality ADD CONSTRAINT speciality_pk PRIMARY KEY ( spec_code );

ALTER TABLE speciality ADD CONSTRAINT spec_desc_unq UNIQUE ( spec_description );

ALTER TABLE adm_prc
    ADD CONSTRAINT admission_admprc FOREIGN KEY ( adm_no )
        REFERENCES admission ( adm_no );

ALTER TABLE item_treatment
    ADD CONSTRAINT admprc_itemtreatment FOREIGN KEY ( adprc_no )
        REFERENCES adm_prc ( adprc_no );

ALTER TABLE item
    ADD CONSTRAINT costcentre_item FOREIGN KEY ( cc_code )
        REFERENCES costcentre ( cc_code );

ALTER TABLE admission
    ADD CONSTRAINT doctor_admission FOREIGN KEY ( doctor_id )
        REFERENCES doctor ( doctor_id );

ALTER TABLE doctor_speciality
    ADD CONSTRAINT doctor_doctorspec FOREIGN KEY ( doctor_id )
        REFERENCES doctor ( doctor_id );

ALTER TABLE adm_prc
    ADD CONSTRAINT doctor_performadmprc FOREIGN KEY ( perform_dr_id )
        REFERENCES doctor ( doctor_id );

ALTER TABLE adm_prc
    ADD CONSTRAINT doctor_requestadmprc FOREIGN KEY ( request_dr_id )
        REFERENCES doctor ( doctor_id );

ALTER TABLE item_treatment
    ADD CONSTRAINT item_itemtreatment FOREIGN KEY ( item_code )
        REFERENCES item ( item_code );

ALTER TABLE admission
    ADD CONSTRAINT patient_admission FOREIGN KEY ( patient_id )
        REFERENCES patient ( patient_id );

ALTER TABLE adm_prc
    ADD CONSTRAINT procedure_admproc FOREIGN KEY ( proc_code )
        REFERENCES procedure ( proc_code );

ALTER TABLE doctor_speciality
    ADD CONSTRAINT spec_doctorspec FOREIGN KEY ( spec_code )
        REFERENCES speciality ( spec_code );

REM INSERTING into PROCEDURE
SET DEFINE OFF;
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (12055,'Appendectomy','Removel of appendix',60,250);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (15509,'X-ray, Right knee','Right knee Bi-Lateral 2D Scan',20,75);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (15510,'X-ray, Left knee','Left knee Bi-Lateral 2D Scan',20,75);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (15511,'MRI','Imaging of brain',90,200);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (17122,'Childbirth','Caesarean section',80,500);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (19887,'Colonoscopy','Bowel examination',25,68);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (23432,'Mental health','Counselling for children',60,98);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (27459,'Corneal replacement','Replacement of cornea',60,633);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (29844,'Tonsillectomy','Removal of tonsils',45,109.28);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (32266,'Hemoglobin concentration','Measuring oxygen carrying protein in blood',15,76);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (33335,'Eye test','Test for eye problems',40,70.45);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (40099,'Scratch test','Allergy test on skin surface',15,40);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (40100,'Skin surgery','Removal of mole',20,33.5);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (43111,'Angiogram','Insertion of catheter into artery',180,355);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (43112,'Thoracic surgery','Removal of lung tumour',180,399);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (43114,'Heart surgery','Insertion of a pacemaker',45,120.66);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (43556,'Vascular surgery','Removel of varicose veins',120,243.1);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (49518,'Total replacement, Right knee','Right knee replacement by artificial joint',180,350);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (54132,'Plastic surgery','Burn surgery to repair skin',170,244);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (65554,'Blood screen','Full blood test',10,30);
Insert into PROCEDURE (PROC_CODE,PROC_NAME,PROC_DESCRIPTION,PROC_TIME,PROC_STD_COST) values (71432,'Genetic testing','Screening for genetically carried diseases',15,65.2);

REM INSERTING into SPECIALITY
SET DEFINE OFF;
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('ALLERG','Allergy and immunology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('CARDIO','Cardiovascular disease');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('DERMAT','Dermatology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('GENETI','Medical genetics');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('HEMATO','Hematology and oncology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('NEUROL','Neurological surgery');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('OBSTET','Obstetrics and gynecology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('OPHTHA','Ophthalmology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('ORTHOP','Orthopedic surgery');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('OTOLAR','Otolaryngology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('PATHOL','Pathology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('PEDIAT','Pediatrics');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('PLASTI','Plastic surgery');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('PULMON','Pulmonary disease and critical care medicine');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('RADIOL','Radiology');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('THORAC','Thoracic surgery');
Insert into SPECIALITY (SPEC_CODE,SPEC_DESCRIPTION) values ('VASCUL','Vascular surgery');

REM INSERTING into DOCTOR
SET DEFINE OFF;
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1005,'Mr','Erich','Argabrite','1755428382');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1012,'Dr','Tedi','Jeeves','9188264756');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1018,'Dr','Caresa','Cornilleau','1334007521');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1027,'Mr','Mikaela','Leyban','9296294312');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1028,'Ms','Cherilyn','Bray','7359457889');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1033,'Dr','Sawyer','Haisell','3914928134');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1048,'Dr','Steffane','Banstead','9466787825');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1056,'Ms','Minnnie','Udey','8158285073');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1060,'Dr','Decca','Blankhorn','4942993995');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1061,'Mr','Jere','Digman','1281091935');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1064,'Mr','Rudolph','Jowett','9873380817');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1069,'Ms','Corry','Walrond','3531087771');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1084,'Ms','Rollie','Whayman','5649708242');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1095,'Dr','Bonnibelle','Misk','1289776540');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1099,'Ms','Irv','Tourner','5689696759');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (1298,'Mr','Graham','Brown','1234567899');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (2459,'Dr','Robert','Lu','1515141312');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (7890,'Dr','Mary','Wei','6655443377');
Insert into DOCTOR (DOCTOR_ID,DOCTOR_TITLE,DOCTOR_FNAME,DOCTOR_LNAME,DOCTOR_PHONE) values (7900,'Dr','Juixan','Wei','6622544311');

REM INSERTING into DOCTOR_SPECIALITY
SET DEFINE OFF;
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('ALLERG',1012);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('CARDIO',1005);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('DERMAT',1028);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('GENETI',1033);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('HEMATO',1061);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('NEUROL',1048);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('OBSTET',1056);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('OPHTHA',1060);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('ORTHOP',1298);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('ORTHOP',2459);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('ORTHOP',7890);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('OTOLAR',1064);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('PATHOL',1061);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('PATHOL',1069);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('PEDIAT',1027);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('PLASTI',1084);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('PLASTI',1095);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('PULMON',1033);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('RADIOL',1060);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('THORAC',1060);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('VASCUL',1005);
Insert into DOCTOR_SPECIALITY (SPEC_CODE,DOCTOR_ID) values ('ORTHOP',7900);


REM INSERTING into COSTCENTRE
SET DEFINE OFF;
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC001','Administration','Alexa Kitson');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC002','Cleaning','Leonid Barlace');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC003','Dietary and Cafeteria','Belinda Domnick');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC004','Ancillary Supplies','Delia Le Brun			');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC005','Operating Theatre','Leigh Lenox');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC006','Anaesthesiology','Bette McLleese');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC007','Labor and Delivery','Shay Upton');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC008','Radiology','Glenine Eymor');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC009','Laboratory Supplies','Francyne Ordidge');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC010','Inhalation Therapy','Fletch Carriage');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC011','Physical Therapy','Talya Townsley');
Insert into COSTCENTRE (CC_CODE,CC_TITLE,CC_MANAGER_NAME) values ('CC012','Pharmacy','Aurelie Clemensen');


REM INSERTING into ITEM
SET DEFINE OFF;
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('NE001','Needle Spinal 22g X 5" Becton Dickinson',20,3.45,'CC004');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('CA002','Catheter i.V. Optiva 22g X 25mm ',50,2.25,'CC004');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('OV001','Interlink Vial Access Cannula ',30,4.28,'CC004');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('TE001','Tube Extension Terumo 75cm ',50,1.72,'CC004');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('AN002','Std Anaesthetic Pack',25,182.33,'CC004');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('SS006','Stainless Steel Pins',100,15.1,'CC004');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('KN056','Right Knee Brace',10,123,'CC004');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('PS318','Pump Suction Askir Liner Disp for cam31843 ',100,4.76,'CC009');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('ST252','Sigmoidoscope Tube Heine Disposable 25s 250x20mm',100,0.72,'CC009');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('AT258','Anoscope Tubes Heine Disposable 25s 85x20mm',50,1.14,'CC009');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('TN010','Thermometer Nextemp Disposable ',500,0.45,'CC009');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('LB250','Laryngoscope Blade Heine Mcintosh F/Opt',50,215.1,'CC009');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('CE001','Chloromycetin Eye Ointment 4g ',20,3.98,'CC012');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('EA030','Epipen Adult 0.30mg. Adrenalin ',20,110.15,'CC012');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('CE010','Chlorsig Eye Oint. 1% ',25,3.98,'CC012');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('AP050','Amethocaine 0.5% 20s Prev Tetracaine 0.5%',25,81.2,'CC012');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('BI500','Bupivacaine Inj .5% 10ml Steriamp ',10,365.48,'CC012');
Insert into ITEM (ITEM_CODE,ITEM_DESCRIPTION,ITEM_STOCK,ITEM_COST,CC_CODE) values ('CF050','Cophenylcaine Forte Nasal Spray 50ml',10,62.04,'CC012');

commit;
