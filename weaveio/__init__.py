from pathlib import Path

from .__version__ import __version__

import logging

try:
    from boxing import boxing
    from colored import fore, style
    from pkg_info import get_pkg_info
    from semver import compare
    import urllib
    import json
    from datetime import datetime
    from typing import Callable


    def parse_changes(description):
        log = description.split('machine-readable-change-log\n###########################\n')[-1]
        return log


    def get_other_descriptions(name, version):
        return json.loads(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=3).read())['info']['description']


    class UpdateNotify(object):
        def __init__(self, name: str, version: str):
            self.name: str = name
            self.version: str = version
            self.last_checked_path = Path(__file__).parent / '.last-checked'
            self.time_fmt = '%Y-%m-%dT%H:%M:%S'
            self._pkg = None

        @property
        def pkg(self):
            if self._pkg is None:
                self._pkg = get_pkg_info(self.name)
            return self._pkg

        def last_checked(self):
            try:
                with open(str(self.last_checked_path), 'r') as f:
                    return datetime.strptime(f.read().strip(), self.time_fmt)
            except FileNotFoundError:
                return None

        def too_soon(self):
            lc = self.last_checked()
            if lc is None:
                return False
            seconds = (datetime.now() - lc).total_seconds()
            return (seconds / 60 / 60) < 6

        def update_last_checked(self):
            with open(str(self.last_checked_path), 'w') as f:
                f.write(datetime.now().strftime(self.time_fmt))

        def is_latest_version(self) -> bool:
            self.latest = self.pkg.version
            return True if compare(self.version, self.latest) >= 0 else False

        def render_changes(self):
            releases = [(k, datetime.strptime(v[0]['upload_time'], self.time_fmt)) for k, v in self.pkg.raw_data['releases'].items()]
            releases.sort(key=lambda x: x[1])
            release_names, _ = zip(*releases)
            changes = [parse_changes(get_other_descriptions(self.pkg.name, v)) for v in release_names]
            newer_changes = changes[release_names.index(self.pkg.version):]
            return '\n'.join(newer_changes), len(newer_changes)

        def notify(self) -> None:
            if self.too_soon():
                logging.info('Skipping version checking since its has not been 6 hours since the last check.')
                return
            self.update_last_checked()
            if self.is_latest_version():
                return
            action, arg = print, self.default_message()
            action(arg) if arg else action()

        def default_message(self) -> str:
            changes, nchanges = self.render_changes()
            version = fore.GREY_53 + self.version + style.RESET
            latest = fore.LIGHT_GREEN + self.latest + style.RESET
            command = fore.LIGHT_BLUE + 'pip install -U ' + self.name + style.RESET
            nchanges = fore.LIGHT_GREEN + str(nchanges) + style.RESET
            return boxing(f'Update available {version} → {latest} ({nchanges} new changes)\n' +
                          f'Run {command} to update\n'
                          f'{changes}')

    UpdateNotify('weaveio', __version__).notify()

except ImportError:
    from warnings import warn
    warn('Please run `pip install boxing colored pkg_info semver` to alert you to updated versions of the weaveio library')
except Exception as e:
    logging.exception('There was a problem in alerting you to updated versions of the weaveio library...', exc_info=True)
