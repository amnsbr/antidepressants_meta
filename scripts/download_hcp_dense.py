from getpass import getpass
import pyxnat
import os

print("Please enter your https://db.humanconnectome.org credentials")
user = input("Username: ")
password = getpass("Password: ")
cdb = pyxnat.Interface("https://db.humanconnectome.org", user, password)
dense_file = (
    cdb.select.project("HCP_Resources")
    .resource("GroupAvg")
    .file("HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.zip")
)
print(dense_file.size())
out_dir = input("Download location: ")
os.makedirs(out_dir, exist_ok=True)
dense_file.get(
    os.path.join(
        out_dir, "HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.zip"
    )
)
