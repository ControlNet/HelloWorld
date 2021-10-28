/* (i)*/
CREATE SEQUENCE patient_seq 
MINVALUE 200000
START WITH 200000
INCREMENT BY 10
CACHE 2;

CREATE SEQUENCE adm_seq
MINVALUE 200000
START WITH 200000
INCREMENT BY 10
CACHE 2;

CREATE SEQUENCE adm_prc_seq
MINVALUE 200000
START WITH 200000
INCREMENT BY 10
CACHE 2;

/* (ii)*/
INSERT INTO patient 
VALUES (patient_seq.NEXTVAL,'Peter','Xiue','14 Narrow Lane Caulfield',TO_DATE('01-Oct-1981','dd-Mon-yyyy'),'0123456789');

INSERT INTO admission (adm_no, adm_date_time, patient_id, doctor_id) 
VALUES (adm_seq.NEXTVAL,TO_DATE('16-May-2019 10:00:00','dd-Mon-yyyy HH24:mi:ss'),patient_seq.CURRVAL, (SELECT doctor_id 
                                                                                                        FROM doctor 
                                                                                                        WHERE doctor_fname = 'Sawyer' 
                                                                                                            AND UPPER(doctor_lname) = 'HAISELL'));
COMMIT;

/* (iii)*/
UPDATE doctor_speciality
SET spec_code = (SELECT spec_code
                    FROM speciality
                    WHERE spec_description = 'Vascular surgery')
WHERE doctor_id = (SELECT doctor_id
                    FROM doctor
                    WHERE doctor_fname = 'Decca'
                    AND UPPER(doctor_lname) = 'BLANKHORN')
    AND spec_code = (SELECT spec_code
                    FROM speciality
                    WHERE spec_description = 'Thoracic surgery');

COMMIT;

/* (iv)*/
DELETE FROM doctor_speciality
WHERE spec_code = (SELECT spec_code 
                    FROM speciality
                    WHERE spec_description = 'Medical genetics');

DELETE FROM speciality
WHERE spec_description = 'Medical genetics';

COMMIT;
