import subprocess

# Get the list of installed packages
installed_packages = subprocess.check_output(['pip', 'freeze']).decode().split('\n')

# Iterate over the installed packages and upgrade each one
for package in installed_packages:
    if package:
        package_name = package.split('==')[0]
        subprocess.call(['pip', 'install', '--upgrade', package_name])
