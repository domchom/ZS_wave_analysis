import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveanalysis import runner


if __name__ == '__main__':
    runner.main()

print('Done with script!')
