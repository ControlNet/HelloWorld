/* (i)*/
SET SERVEROUTPUT ON;

CREATE OR REPLACE TRIGGER item_code_change_message
AFTER UPDATE OF item_code ON item
FOR EACH ROW
BEGIN
    dbms_output.put_line('change '||:old.item_code||' for the "'||:old.item_description||'" to '||:new.item_code);
END;
/

/* (ii)*/
CREATE OR REPLACE TRIGGER name_null_check
BEFORE INSERT OR UPDATE OF patient_fname, patient_lname ON patient
FOR EACH ROW
BEGIN
    IF :new.patient_fname IS NULL AND :new.patient_lname IS NULL THEN
        raise_application_error(-20000, 'Patient first name and last name cannot both be null.');
    END IF;
END;
/

/* (iii)*/
CREATE OR REPLACE TRIGGER maintain_item_stock
AFTER INSERT ON item_treatment
FOR EACH ROW
BEGIN
    UPDATE item
    SET item_stock = item_stock - :new.it_qty_used
    WHERE item_code = :new.item_code;
END;
/
