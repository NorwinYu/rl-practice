from setuptools import setup, find_packages

setup(
    name="rl2020",
    version="0.1",
    description="Reinforcement Learning in UoE (HW)",
    # author="Filippos Christianos",
    url="https://github.com/semitable/uoe-rl2020",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy>=1.18",
        "torch>=1.3" "gym>=0.12",
        "gym[box2d]",
        "tqdm>=4.41",
        "pyglet>=1.3",
        "matplotlib>=3.1",
        "pytest>=5.3",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
