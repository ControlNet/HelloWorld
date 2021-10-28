/* (i)*/
ALTER TABLE item 
    ADD re_order_level number(3,0);

COMMENT ON COLUMN item.re_order_level IS 
    'The re-order level of this item.';
    
UPDATE item SET re_order_level = item_stock / 2;

ALTER TABLE item
    MODIFY re_order_level NOT NULL; 

COMMIT;
  
/* (ii)*/

-- Create new bridge entities named doc_perform and perform_role

CREATE TABLE doc_perform (
    adprc_no            NUMBER(7) NOT NULL,
    doctor_id           NUMBER(4) NOT NULL,
    role_no             NUMBER(1) NOT NULL
);

COMMENT ON COLUMN doc_perform.adprc_no IS
    'Admission number (PK)';
COMMENT ON COLUMN doc_perform.doctor_id IS
    'Doctor id (PK)';
COMMENT ON COLUMN doc_perform.role_no IS
    'Role number';

ALTER TABLE doc_perform ADD CONSTRAINT doc_perform_pk PRIMARY KEY (adprc_no, doctor_id);

CREATE TABLE perform_role (
    role_no             NUMBER(1) NOT NULL,
    role_desc           VARCHAR(50) NOT NULL
);

COMMENT ON COLUMN perform_role.role_no IS
    'Role number (PK)';
COMMENT ON COLUMN perform_role.role_desc IS
    'Role description';

ALTER TABLE perform_role ADD CONSTRAINT perform_role_pk PRIMARY KEY (role_no);

ALTER TABLE doc_perform ADD CONSTRAINT docperform_admprc_fk FOREIGN KEY (adprc_no) REFERENCES adm_prc (adprc_no);

ALTER TABLE doc_perform ADD CONSTRAINT docperform_doctor_fk FOREIGN KEY (doctor_id) REFERENCES doctor (doctor_id);

ALTER TABLE doc_perform ADD CONSTRAINT docperform_performrole_fk FOREIGN KEY (role_no) REFERENCES perform_role (role_no);

-- Insert necessary role data into table perform_role 
INSERT INTO perform_role VALUES (1,'leader');
INSERT INTO perform_role VALUES (2,'ancillary');

-- Insert the original data in adm_prc into doc_perform
INSERT INTO doc_perform (SELECT adprc_no, perform_dr_id, 1
                        FROM adm_prc
                        WHERE perform_dr_id IS NOT NULL);
                        
-- Drop unnecessary attribute perform_dr_id in table adm_prc
ALTER TABLE adm_prc DROP COLUMN perform_dr_id;
                        
-- Add a doctor 1060 as an Ancillary doctor into adprc_no 118149
INSERT INTO doc_perform VALUES (118149, 1060, (SELECT role_no FROM perform_role WHERE LOWER(role_desc) = 'ancillary'));

COMMIT;

