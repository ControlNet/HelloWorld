/* (i)*/
SELECT d.doctor_title, d.doctor_fname, d.doctor_lname, d.doctor_phone
FROM doctor d JOIN doctor_speciality ds ON d.doctor_id = ds.doctor_id
WHERE ds.spec_code = (SELECT spec_code 
                        FROM speciality 
                        WHERE UPPER(spec_description) = 'ORTHOPEDIC SURGERY')
ORDER BY doctor_lname, doctor_fname;

/* (ii)*/
SELECT i.item_code, i.item_description, i.item_stock, c.cc_title
FROM item i JOIN costcentre c ON i.cc_code = c.cc_code
WHERE i.item_stock > 50
    AND LOWER(i.item_description) LIKE '%disposable%'
ORDER BY i.item_code;
    
/* (iii)*/
SELECT p.patient_id, p.patient_fname||' '||p.patient_lname AS "Patient Name", to_char(a.adm_date_time,'dd-Mon-yyyy hh24:mi') AS ADMDATETIME, d.doctor_title||' '||d.doctor_fname||' '||d.doctor_lname AS "Doctor Name"
FROM (patient p JOIN admission a ON p.patient_id = a.patient_id) 
    JOIN doctor d ON a.doctor_id = d.doctor_id
WHERE to_char(a.adm_date_time,'dd-Mon-yyyy') = '14-Mar-2019'
ORDER BY a.adm_date_time;

/* (iv)*/
SELECT proc_code, proc_name, proc_description, to_char(proc_std_cost,'$990.99') AS "Procedure Standard Cost"
FROM procedure
WHERE proc_std_cost < (SELECT avg(proc_std_cost)
                        FROM procedure)
ORDER BY proc_std_cost DESC;
 
/* (v)*/
SELECT p.patient_id, p.patient_lname, p.patient_fname, to_char(p.patient_dob,'dd-Mon-yyyy') AS DOB, COUNT(*) AS numberadmissions
FROM patient p JOIN admission a ON p.patient_id = a.patient_id
GROUP BY p.patient_id, p.patient_lname, p.patient_fname, p.patient_dob
HAVING COUNT(*) > 2
ORDER BY COUNT(*) DESC, p.patient_dob;
    
/* (vi)*/
SELECT a.adm_no, a.patient_id, p.patient_fname, p.patient_lname, to_char(trunc(to_number(a.adm_discharge - a.adm_date_time)),'90')||' days '||to_char(mod(to_number(a.adm_discharge - a.adm_date_time),1)*24,'90.9')||' hrs' AS staylength
FROM admission a JOIN patient p ON a.patient_id = p.patient_id
WHERE a.adm_discharge IS NOT NULL
    AND a.adm_discharge - a.adm_date_time > (SELECT avg(adm_discharge - adm_date_time)
                                            FROM admission)
ORDER BY a.adm_no;
    
/* (vii)*/
SELECT p.proc_code, p.proc_name, p.proc_description, p.proc_time, to_char((avg(ap.adprc_pat_cost) - avg(p.proc_std_cost)),'9990.99') AS "Price Differential"
FROM procedure p JOIN adm_prc ap ON p.proc_code = ap.proc_code
GROUP BY p.proc_code, p.proc_name, p.proc_description, p.proc_time
ORDER BY p.proc_code;

/* (viii)*/
SELECT p.proc_code, p.proc_name, nvl(i.item_code,'---') AS item_code, nvl(i.item_description,'---') AS item_description, nvl(to_char(max(it.it_qty_used)),'---') AS max_qty_used
FROM ((procedure p LEFT JOIN adm_prc ap ON p.proc_code = ap.proc_code)
    LEFT JOIN item_treatment it ON ap.adprc_no = it.adprc_no)
    LEFT JOIN item i ON it.item_code = i.item_code
GROUP BY p.proc_code, p.proc_name, nvl(i.item_code,'---'), nvl(i.item_description,'---')
ORDER BY p.proc_name;

