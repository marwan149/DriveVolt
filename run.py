from pathlib import Path
import sys

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent
    src_path = project_root / 'src'
    sys.path.insert(0, str(src_path))

    from inverter_model.main import main

    main()
