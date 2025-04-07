from setuptools import setup, find_packages

HYPHEN_E_DASH = '-e .'
def remove_hyphen_e(requirements):
    """
    This function removes '-e .' from the requirements list.
    """
    return [req.replace(HYPHEN_E_DASH, '') for req in requirements if req.strip() != HYPHEN_E_DASH]
def get_requirements(file_path):
    """
    This function returns a list of requirements from the given file path.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        requirements = remove_hyphen_e(requirements)
        return requirements

setup(
    name='mlproject_08_04',
    version='0.1',
    author='sumanjit moshat',
    author_email='moshatsumanjit94@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

