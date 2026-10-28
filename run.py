#!/usr/bin/env python
"""
ExoMiner Unified Project Runner
===============================
Main entry point for running the ExoMiner project on Windows and other platforms.

Features:
  1. Run ExoMiner Pipeline (data downloading, TCE extraction, light curve & difference
     image preprocessing, TFRecord dataset generation, and ExoMiner++ neural net inference).
  2. Run Interactive Vetting Catalog Web App (Dash web application for exploring TCEs).
  3. Run ExoplanetAI Streamlit Web App (if available).
  4. Dependency & Environment Diagnostics (checks Python, Conda environment, TensorFlow,
     MAST access, and model weights).

Usage Examples:
  python run.py                               # Interactive menu mode
  python run.py pipeline                      # Run pipeline with default inputs
  python run.py pipeline --input inputs/input_tics.csv --output output/run_result
  python run.py webapp                        # Launch Dash vetting catalog web app
  python run.py webapp --port 8050            # Specify custom port
  python run.py check                         # Run environment health check
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import argparse

# Suppress TensorFlow & package warning noise on startup
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Environment & Python Interpreter Verification / Auto-reexec
# -----------------------------------------------------------------------------
EXPECTED_CONDA_ENV = "exoplnt_dl"
CONDA_PYTHON_CANDIDATES = [
    Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "envs" / EXPECTED_CONDA_ENV / "python.exe",
    Path(os.environ.get("CONDA_PREFIX", "")) / "envs" / EXPECTED_CONDA_ENV / "python.exe",
    Path("C:/Users/anand/anaconda3/envs/exoplnt_dl/python.exe"),
]

def ensure_compatible_python():
    """
    Ensure the script is running with the 'exoplnt_dl' conda environment
    where TensorFlow, Dash, Astropy, etc. are all installed.
    """
    # If already running inside exoplnt_dl, nothing to do
    if "exoplnt_dl" in sys.prefix.lower() or os.environ.get("CONDA_DEFAULT_ENV") == EXPECTED_CONDA_ENV:
        return

    # Look for exoplnt_dl python executable and switch to it
    for candidate in CONDA_PYTHON_CANDIDATES:
        if candidate.is_file():
            if Path(sys.executable).resolve() != candidate.resolve():
                print(f"[*] Activating '{EXPECTED_CONDA_ENV}' environment: {candidate}")
                cmd = [str(candidate), __file__] + sys.argv[1:]
                result = subprocess.call(cmd)
                sys.exit(result)
            return

    # Fallback: check if current environment has required packages
    try:
        importlib.import_module("tensorflow")
        importlib.import_module("dash")
        return
    except (ImportError, Exception):
        print(f"[!] Warning: Not running in '{EXPECTED_CONDA_ENV}'. Some dependencies may be missing.")
        print(f"    Please run: conda activate {EXPECTED_CONDA_ENV}")

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent

def locate_exominer_dir() -> Path:
    """Locate the ExoMiner root directory containing exominer_pipeline."""
    candidates = [
        WORKSPACE_ROOT / "ExoMiner",
        WORKSPACE_ROOT / "ExoMiner-main",
        WORKSPACE_ROOT,
    ]
    for candidate in candidates:
        if (candidate / "exominer_pipeline" / "run_pipeline.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate ExoMiner source directory with 'exominer_pipeline'.")

EXOMINER_DIR = locate_exominer_dir()

# Add to sys.path
if str(EXOMINER_DIR) not in sys.path:
    sys.path.insert(0, str(EXOMINER_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


# -----------------------------------------------------------------------------
# Component: Pipeline Runner
# -----------------------------------------------------------------------------
def run_pipeline(
    input_csv: str = None,
    output_dir: str = None,
    mode: str = "2min",
    num_processes: int = 1,
    num_jobs: int = 1,
    download_spoc_products: bool = False,
    model: str = "exominer++_single",
    ruwe_source: str = "gaiadr2",
    stellar_source: str = "ticv8",
):
    """Executes the ExoMiner pipeline with safe defaults."""
    print("=" * 65)
    print("           STARTING EXOMINER TRANSIT VETTING PIPELINE")
    print("=" * 65)

    # Determine input CSV
    if not input_csv:
        potential_inputs = [
            WORKSPACE_ROOT / "inputs" / "input_tics.csv",
            WORKSPACE_ROOT / "inputs" / "tics_tbl.csv",
            EXOMINER_DIR / "inputs" / "input_tics.csv",
            EXOMINER_DIR / "inputs" / "tics_tbl.csv",
            EXOMINER_DIR / "tics.csv",
        ]
        for candidate in potential_inputs:
            if candidate.exists():
                input_csv = str(candidate.resolve())
                break

    if not input_csv or not Path(input_csv).exists():
        print(f"[!] Error: Input CSV not found: {input_csv}")
        return False

    # Determine output directory
    if not output_dir:
        output_dir = str((WORKSPACE_ROOT / "output").resolve())

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[*] Input TIC table : {input_csv}")
    print(f"[*] Output directory: {output_dir}")
    print(f"[*] Collection mode : {mode}")
    print(f"[*] Processes / Jobs: {num_processes} process(es), {num_jobs} job(s)")
    print(f"[*] ExoMiner Model  : {model}")
    print("-" * 65)

    # Set working directory to EXOMINER_DIR so relative config paths work
    original_cwd = os.getcwd()
    os.chdir(EXOMINER_DIR)

    try:
        pipeline_mod = importlib.import_module("exominer_pipeline.run_pipeline")
        run_exominer_pipeline_main = getattr(pipeline_mod, "run_exominer_pipeline_main")

        run_exominer_pipeline_main(
            output_dir=output_dir,
            tic_ids_fp=input_csv,
            data_collection_mode=mode,
            tic_ids=None,
            num_processes=num_processes,
            num_jobs=num_jobs,
            download_spoc_data_products="true" if download_spoc_products else "false",
            external_data_repository="null",
            stellar_parameters_source=stellar_source,
            ruwe_source=ruwe_source,
            exominer_model=model,
        )
        print("\n[+] ExoMiner Pipeline finished successfully!")
        print(f"[+] Output files saved in: {output_dir}")
        return True
    except Exception as e:
        print(f"\n[!] Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_cwd)


# -----------------------------------------------------------------------------
# Component: Dash Web App Runner
# -----------------------------------------------------------------------------
def run_webapp(port: int = 8050, host: str = "127.0.0.1", debug: bool = False):
    """Executes the Dash interactive vetting catalog web application."""
    print("=" * 65)
    print("      STARTING EXOMINER INTERACTIVE VETTING CATALOG (DASH)")
    print("=" * 65)

    app_script = EXOMINER_DIR / "exominer_vetting_pc_catalog_dash-render-web-app" / "vetting_tce_catalog_exominer_dash_app.py"
    if not app_script.exists():
        print(f"[!] Error: Web application script not found at {app_script}")
        return False

    print(f"[*] Serving on http://{host}:{port}/")
    print("[*] Press Ctrl+C to stop the web server.\n")

    # Set working directory to EXOMINER_DIR so relative assets and CSVs load properly
    original_cwd = os.getcwd()
    os.chdir(EXOMINER_DIR)

    try:
        importlib.import_module("dash")
        # Run via python subprocess in the same environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(EXOMINER_DIR)
        cmd = [sys.executable, str(app_script)]
        subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        print("\n[*] Web application stopped.")
    finally:
        os.chdir(original_cwd)


# -----------------------------------------------------------------------------
# Component: Streamlit App Runner
# -----------------------------------------------------------------------------
def run_streamlit(port: int = 8501):
    """Runs the ExoplanetAI Streamlit application with the correct Python interpreter."""
    app_script = EXOMINER_DIR / "ExoplanetAI" / "app.py"
    if not app_script.exists():
        print(f"[!] Error: ExoplanetAI app not found at {app_script}")
        return False

    print("=" * 65)
    print("        STARTING EXOPLANETAI FRONTEND WEB APP (STREAMLIT)")
    print("=" * 65)

    # Search for an environment that has both streamlit and streamlit_option_menu
    streamlit_python = None
    candidate_pythons = [
        Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "envs" / "exoplanet" / "python.exe",
        Path("C:/Users/anand/anaconda3/envs/exoplanet/python.exe"),
        Path(sys.executable),
        Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "envs" / EXPECTED_CONDA_ENV / "python.exe",
    ]

    for cand in candidate_pythons:
        if cand.is_file():
            test_res = subprocess.call(
                [str(cand), "-c", "import streamlit, streamlit_option_menu"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if test_res == 0:
                streamlit_python = str(cand)
                break

    if not streamlit_python:
        print("[!] Missing required packages ('streamlit' or 'streamlit_option_menu').")
        print("    Installing streamlit-option-menu...")
        subprocess.call([sys.executable, "-m", "pip", "install", "streamlit-option-menu"])
        streamlit_python = sys.executable

    print(f"[*] Serving ExoplanetAI UI on http://localhost:{port}")
    print("[*] Opening in your web browser...")
    cmd = [streamlit_python, "-m", "streamlit", "run", str(app_script), "--server.port", str(port)]
    subprocess.call(cmd, cwd=str(EXOMINER_DIR / "ExoplanetAI"))


# -----------------------------------------------------------------------------
# Component: Diagnostics & Health Check
# -----------------------------------------------------------------------------
def run_diagnostics():
    """Verifies all required libraries, files, and configurations."""
    print("=" * 65)
    print("             EXOMINER ENVIRONMENT & SYSTEM CHECK")
    print("=" * 65)

    print(f"[*] Python Executable : {sys.executable}")
    print(f"[*] Python Version    : {sys.version.split()[0]}")
    print(f"[*] Workspace Root    : {WORKSPACE_ROOT}")
    print(f"[*] ExoMiner Dir      : {EXOMINER_DIR}")
    print("-" * 65)

    # 1. Package checks
    packages = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("astropy", "Astropy"),
        ("lightkurve", "Lightkurve"),
        ("yaml", "PyYAML"),
        ("dash", "Dash"),
        ("mpi4py", "MPI4Py"),
        ("astroquery", "Astroquery"),
        ("streamlit", "Streamlit (Optional)"),
    ]

    print("[*] Checking Python packages:")
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "OK")
            print(f"    [+] {label:<22}: Found (v{ver})")
        except ImportError:
            is_opt = "Optional" in label
            tag = "[-]" if is_opt else "[!]"
            status = "Not installed (Optional)" if is_opt else "MISSING"
            print(f"    {tag} {label:<22}: {status}")

    print("-" * 65)
    # 2. Key files check
    model_fp = EXOMINER_DIR / "exominer_pipeline" / "data" / "exominer-plusplus_cv-iter0-model0_tess-spoc-2min-s1s67_tess-kepler.keras"
    catalog_fp = EXOMINER_DIR / "exominer_vetting_pc_catalog_dash-render-web-app" / "data" / "exominer_vetting_tess-spoc-2-min-s1s67_dashtable_dvm-url_scoregt0.1.csv"
    input_csv = WORKSPACE_ROOT / "inputs" / "input_tics.csv"

    print("[*] Checking Key Assets:")
    print(f"    {'Model Weights':<22}: {'[+] Ready' if model_fp.exists() else '[!] Missing'} ({model_fp.name})")
    print(f"    {'Vetting Catalog':<22}: {'[+] Ready' if catalog_fp.exists() else '[!] Missing'} ({catalog_fp.name})")
    print(f"    {'Input TICs Table':<22}: {'[+] Ready' if input_csv.exists() else '[!] Missing'} ({input_csv.name})")

    print("=" * 65)
    print("[+] System check complete.")


# -----------------------------------------------------------------------------
# Interactive Menu Mode
# -----------------------------------------------------------------------------
def interactive_menu():
    """Displays an interactive console menu to choose what to run."""
    while True:
        print("\n" + "=" * 65)
        print("          EXOMINER - NASA EXOPLANET TRANSIT CLASSIFIER")
        print("=" * 65)
        print("  1) Run ExoMiner Pipeline (TIC ID vetting & inference)")
        print("  2) Run ExoplanetAI Web Frontend (Streamlit Dashboard & UI)")
        print("  3) Run Raw Vetting Catalog Table (Dash Data Table)")
        print("  4) Run System Diagnostics & Environment Check")
        print("  5) Exit")
        print("=" * 65)

        try:
            choice = input("Enter selection [1-5] (default=2): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "1":
            default_input = str((WORKSPACE_ROOT / "inputs" / "input_tics.csv").resolve())
            default_output = str((WORKSPACE_ROOT / "output").resolve())
            print(f"\nPipeline Settings:")
            user_in = input(f"  Input CSV [{default_input}]: ").strip() or default_input
            user_out = input(f"  Output Dir [{default_output}]: ").strip() or default_output
            run_pipeline(input_csv=user_in, output_dir=user_out, num_processes=1, num_jobs=1)
            break
        elif not choice or choice == "2":
            run_streamlit()
            break
        elif choice == "3":
            run_webapp()
            break
        elif choice == "4":
            run_diagnostics()
        elif choice == "5":
            print("Exiting ExoMiner Runner.")
            break
        else:
            print("[!] Invalid option. Please enter a number between 1 and 5.")


# -----------------------------------------------------------------------------
# Main CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    ensure_compatible_python()

    parser = argparse.ArgumentParser(
        description="ExoMiner Unified Runner - NASA Exoplanet Transit Classifier",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the ExoMiner transit vetting pipeline")
    pipeline_parser.add_argument("--input", "-i", dest="input_csv", type=str, default=None,
                                 help="Path to TIC IDs CSV table (columns: tic_id, sector_run)")
    pipeline_parser.add_argument("--output", "-o", dest="output_dir", type=str, default=None,
                                 help="Directory where output results and predictions will be stored")
    pipeline_parser.add_argument("--mode", "-m", dest="mode", type=str, default="2min", choices=["2min", "ffi"],
                                 help="TESS SPOC data collection mode (default: 2min)")
    pipeline_parser.add_argument("--processes", "-p", dest="num_processes", type=int, default=1,
                                 help="Number of parallel processes (default: 1)")
    pipeline_parser.add_argument("--jobs", "-j", dest="num_jobs", type=int, default=1,
                                 help="Number of jobs to partition TIC IDs (default: 1)")
    pipeline_parser.add_argument("--model", type=str, default="exominer++_single",
                                 help="ExoMiner model to use (default: exominer++_single)")
    pipeline_parser.add_argument("--download-spoc-reports", action="store_true",
                                 help="Download links to SPOC DV reports")

    # Frontend / Streamlit command
    frontend_parser = subparsers.add_parser("frontend", help="Run the ExoplanetAI Streamlit frontend web app")
    frontend_parser.add_argument("--port", type=int, default=8501, help="Port to listen on (default: 8501)")

    streamlit_parser = subparsers.add_parser("streamlit", help="Run the ExoplanetAI Streamlit app (alias for frontend)")
    streamlit_parser.add_argument("--port", type=int, default=8501, help="Port to listen on (default: 8501)")

    # Raw Catalog Webapp command
    webapp_parser = subparsers.add_parser("webapp", help="Run the Dash raw vetting catalog table")
    webapp_parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default: 8050)")
    webapp_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")

    # Diagnostics command
    subparsers.add_parser("check", help="Check system dependencies, models, and input files")

    args = parser.parse_args()

    if args.command is None:
        if sys.stdin and sys.stdin.isatty():
            interactive_menu()
        else:
            print("=" * 65)
            print("          EXOMINER - NASA EXOPLANET TRANSIT CLASSIFIER")
            print("=" * 65)
            print("[*] Non-interactive environment detected (e.g. VS Code Output tab).")
            print("[*] Running system check and dependency validation:\n")
            run_diagnostics()
            print("\n" + "=" * 65)
            print("[TIP] To interactively run the pipeline or webapp, open a Terminal:")
            print("        python run.py            (interactive menu)")
            print("        python run.py frontend   (launch ExoplanetAI web app)")
            print("        python run.py pipeline   (run vetting pipeline)")
            print("        python run.py webapp     (launch Dash data table)")
            print("=" * 65)
    elif args.command == "pipeline":
        run_pipeline(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            mode=args.mode,
            num_processes=args.num_processes,
            num_jobs=args.num_jobs,
            download_spoc_products=args.download_spoc_reports,
            model=args.model,
        )
    elif args.command in ("frontend", "streamlit"):
        run_streamlit(port=args.port)
    elif args.command == "webapp":
        run_webapp(port=args.port, host=args.host)
    elif args.command == "check":
        run_diagnostics()


if __name__ == "__main__":
    main()
