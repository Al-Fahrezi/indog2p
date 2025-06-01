from setuptools import setup, find_packages

setup(
    name="indog2p",
    version="0.1.0",
    description="Grapheme-to-Phoneme (G2P) Bahasa Indonesia berbasis BERT + IPA output",
    author="Nama Kamu",
    author_email="email@kamu.com",
    packages=find_packages(exclude=["tests", "notebooks"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "transformers>=4.26",
        "pandas",
        "numpy",
        "pyyaml",
        "scikit-learn",
        "num2words",
        "matplotlib",
        "tqdm"
    ],
    extras_require={
        "notebook": ["notebook", "ipykernel"],
        "onnx": ["onnx", "onnxruntime"],
        "test": ["pytest"]
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            # (opsional) script CLI jika ingin
            # "g2p-predict = scripts.predict:main"
        ]
    }
)
