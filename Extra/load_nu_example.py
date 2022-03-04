def load_nu(error_type = "MSE+MAE", num_patients = 3, nu_file_path = "../nu_patients.txt"):
    error_dict = {"MSE":0,"MAE":1,"MSE+MAE":2}
    nu_dict = {}
    with open(nu_file_path) as nu_file:
        start_line = (num_patients+1)*error_dict[error_type]+ 1
        nu_lines = nu_file.readlines()[start_line:start_line+num_patients]
        for i, patient_line in enumerate(nu_lines):
            patient_line = patient_line[11:]
            nu_dict[i+1] = float(patient_line.strip())
    return nu_dict
nu_dict = load_nu()
print(nu_dict)