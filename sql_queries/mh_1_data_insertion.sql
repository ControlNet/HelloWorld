-- Insert data to table PATIENT
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (136250, 'Marius', 'Everson', '724 Brown Trail', to_date('16-May-1985','dd-Mon-yyyy'), '0262684986');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (130058, 'Betsey', 'Imore', '89975 Nobel Center', to_date('31-Oct-1981','dd-Mon-yyyy'), '0844809666');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (192391, 'Frederich', 'Raymond', '43 Crest Line Road', to_date('01-May-1993','dd-Mon-yyyy'), '0084344917');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (141745, 'Allison', 'Bestiman', '77011 Sycamore Park', to_date('01-Mar-1972','dd-Mon-yyyy'), '0848377906');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (142556, 'Mirna', 'Tonkinson', '1 Clarendon Drive', to_date('28-May-1977','dd-Mon-yyyy'), '0192312802');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (128439, 'Sayre', 'Gatchell', '6244 Kipling Circle', to_date('20-Feb-1972','dd-Mon-yyyy'), '0057906371');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (126462, 'Torey', 'Manueli', '428 Surrey Crossing', to_date('07-Apr-1985','dd-Mon-yyyy'), '0582496317');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (162784, 'Amalita', 'Iannazzi', '95 Union Crossing', to_date('30-Nov-1999','dd-Mon-yyyy'), '0682812097');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (101896, 'Danell', 'Bullard', '00 Russell Center', to_date('30-Mar-1993','dd-Mon-yyyy'), '0872654072');
insert into PATIENT (patient_id, patient_fname, patient_lname, patient_address, patient_dob, patient_contact_phn) values (184787, 'Eben', 'Emblem', '163 Eagle Crest Junction', to_date('03-Feb-1987','dd-Mon-yyyy'), '0591330957');

-- Insert data to table ADMISSION
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (164341, to_date('14-Mar-2019 17:14:26','dd-Mon-yyyy HH24:mi:ss'), to_date('25-Mar-2019 22:36:30','dd-Mon-yyyy HH24:mi:ss'), 136250, 1005);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (169446, to_date('30-Mar-2019 13:18:19','dd-Mon-yyyy HH24:mi:ss'), to_date('02-Apr-2019 05:10:13','dd-Mon-yyyy HH24:mi:ss'), 130058, 1012);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (185179, to_date('16-Feb-2019 21:37:08','dd-Mon-yyyy HH24:mi:ss'), to_date('20-Feb-2019 05:44:02','dd-Mon-yyyy HH24:mi:ss'), 192391, 1018);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (188095, to_date('14-Mar-2019 04:34:47','dd-Mon-yyyy HH24:mi:ss'), to_date('25-Mar-2019 22:46:00','dd-Mon-yyyy HH24:mi:ss'), 141745, 1027);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (125214, to_date('16-Apr-2019 04:51:42','dd-Mon-yyyy HH24:mi:ss'), to_date('20-Apr-2019 01:31:07','dd-Mon-yyyy HH24:mi:ss'), 142556, 1028);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (102461, to_date('18-Apr-2019 05:30:58','dd-Mon-yyyy HH24:mi:ss'), to_date('21-Apr-2019 23:46:32','dd-Mon-yyyy HH24:mi:ss'), 128439, 1033);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (177729, to_date('24-Mar-2019 03:09:46','dd-Mon-yyyy HH24:mi:ss'), to_date('30-Mar-2019 02:22:40','dd-Mon-yyyy HH24:mi:ss'), 126462, 1064);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (168099, to_date('03-Jan-2019 21:52:57','dd-Mon-yyyy HH24:mi:ss'), to_date('09-Jan-2019 23:21:24','dd-Mon-yyyy HH24:mi:ss'), 162784, 1084);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (199883, to_date('30-Apr-2019 03:41:42','dd-Mon-yyyy HH24:mi:ss'), to_date('12-May-2019 11:33:04','dd-Mon-yyyy HH24:mi:ss'), 101896, 7890);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (145205, to_date('05-May-2019 13:55:33','dd-Mon-yyyy HH24:mi:ss'), to_date('11-May-2019 11:53:46','dd-Mon-yyyy HH24:mi:ss'), 184787, 1027);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (169956, to_date('23-Jan-2019 21:47:27','dd-Mon-yyyy HH24:mi:ss'), to_date('29-Jan-2019 21:22:47','dd-Mon-yyyy HH24:mi:ss'), 136250, 1033);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (111097, to_date('04-Feb-2019 16:34:15','dd-Mon-yyyy HH24:mi:ss'), to_date('14-Feb-2019 20:54:11','dd-Mon-yyyy HH24:mi:ss'), 130058, 1033);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (196028, to_date('07-Feb-2019 19:04:58','dd-Mon-yyyy HH24:mi:ss'), to_date('11-Feb-2019 09:39:09','dd-Mon-yyyy HH24:mi:ss'), 136250, 1005);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (131907, to_date('19-Jan-2019 01:07:25','dd-Mon-yyyy HH24:mi:ss'), to_date('22-Jan-2019 04:32:01','dd-Mon-yyyy HH24:mi:ss'), 192391, 1099);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (154252, to_date('21-Apr-2019 12:17:51','dd-Mon-yyyy HH24:mi:ss'), to_date('28-Apr-2019 03:01:37','dd-Mon-yyyy HH24:mi:ss'), 136250, 2459);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (111225, to_date('04-May-2019 16:34:15','dd-Mon-yyyy HH24:mi:ss'), to_date('14-May-2019 20:54:11','dd-Mon-yyyy HH24:mi:ss'), 130058, 1027);
insert into ADMISSION (adm_no, adm_date_time, adm_discharge, patient_id, doctor_id) values (188963, to_date('19-Feb-2019 01:07:25','dd-Mon-yyyy HH24:mi:ss'), to_date('22-Feb-2019 04:32:01','dd-Mon-yyyy HH24:mi:ss'), 192391, 1099);

-- Insert data to table ADM_PRC
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (118149, to_date('15-Mar-2019 16:12:15','dd-Mon-yyyy HH24:mi:ss'),260,122.69,164341,12055,1012,1027);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (199563, to_date('01-Apr-2019 15:46:48','dd-Mon-yyyy HH24:mi:ss'),210,0,169446,43556,1028,1028);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (111762, to_date('17-Feb-2019 23:22:14','dd-Mon-yyyy HH24:mi:ss'),30,0,185179,40100,1064,1099);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id) values (113321, to_date('16-Mar-2019 14:33:50','dd-Mon-yyyy HH24:mi:ss'),105,0,188095,23432,1069);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (184226, to_date('19-Apr-2019 14:31:37','dd-Mon-yyyy HH24:mi:ss'),62,0.9,102461,71432,1084,7900);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (136544, to_date('26-Mar-2019 04:13:21','dd-Mon-yyyy HH24:mi:ss'),35,22.5,177729,40100,1095,2459);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (144370, to_date('01-May-2019 23:35:26','dd-Mon-yyyy HH24:mi:ss'),118,0,199883,23432,2459,2459);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (104918, to_date('08-Feb-2019 01:29:03','dd-Mon-yyyy HH24:mi:ss'),350,0,111097,43111,1099,7900);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id) values (110033, to_date('22-Apr-2019 06:02:42','dd-Mon-yyyy HH24:mi:ss'),401.5,0,154252,43112,7890);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (195300, to_date('04-Jan-2019 15:15:10','dd-Mon-yyyy HH24:mi:ss'),248,1827.4,168099,54132,1018,1298);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id) values (116868, to_date('06-Jan-2019 21:11:50','dd-Mon-yyyy HH24:mi:ss'),201,552.19,168099,15511,1027);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id) values (192689, to_date('24-Apr-2019 07:08:55','dd-Mon-yyyy HH24:mi:ss'),115,0,154252,29844,1028);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id) values (186691, to_date('09-Feb-2019 04:03:49','dd-Mon-yyyy HH24:mi:ss'),378,0,111097,43112,1033);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (105987, to_date('02-May-2019 13:30:11','dd-Mon-yyyy HH24:mi:ss'),400,730.96,199883,43112,1060,1033);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (189196, to_date('16-Mar-2019 14:10:35','dd-Mon-yyyy HH24:mi:ss'),245,19.9,164341,43556,1064,2459);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (138485, to_date('17-Mar-2019 08:25:40','dd-Mon-yyyy HH24:mi:ss'),103,0,188095,23432,1069,1056);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (165766, to_date('18-Mar-2019 23:51:33','dd-Mon-yyyy HH24:mi:ss'),98,0,188095,23432,1084,1060);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (154673, to_date('27-Apr-2019 04:50:02','dd-Mon-yyyy HH24:mi:ss'),25,54.1,154252,40100,1095,7900);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (108470, to_date('03-May-2019 19:09:38','dd-Mon-yyyy HH24:mi:ss'),356,0,199883,43111,1099,7890);
insert into ADM_PRC (adprc_no, adprc_date_time, adprc_pat_cost, adprc_items_cost, adm_no, proc_code, request_dr_id, perform_dr_id) values (148385, to_date('04-May-2019 08:06:31','dd-Mon-yyyy HH24:mi:ss'),240,11.25,199883,43556,1298,2459);

-- Insert data to table ITEM_TREATMENT
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (118149,'NE001',5,17.25);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (118149,'TN010',8,3.6);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (118149,'CE001',10,39.8);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (118149,'CF050',1,62.04);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (184226,'TN010',2,0.9);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (136544,'CA002',10,22.5);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (195300,'BI500',5,1827.4);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (116868,'ST252',2,1.44);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (116868,'EA030',5,550.75);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (105987,'BI500',2,730.96);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (189196,'CE010',5,19.9);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (154673,'TE001',10,17.2);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (154673,'ST252',20,14.4);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (154673,'TN010',50,22.5);
insert into ITEM_TREATMENT (adprc_no, item_code, it_qty_used, it_item_total_cost) values (148385,'CA002',5,11.25);

commit;