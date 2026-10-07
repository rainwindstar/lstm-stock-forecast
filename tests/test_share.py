import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import start_share as share


class SharingTests(unittest.TestCase):
    def test_cloudflare_output_and_failure(self):
        proc = Mock(stdout=io.StringIO('request https://api.trycloudflare.com failed\nhttps://sample-name.trycloudflare.com ready\n'))
        self.assertEqual(share.cloudflare_url(proc), 'https://sample-name.trycloudflare.com')
        with self.assertRaises(RuntimeError):
            share.cloudflare_url(Mock(stdout=io.StringIO('request https://api.trycloudflare.com failed\n')))
        with self.assertRaises(RuntimeError):
            share.cloudflare_url(proc, timeout=0)

    def test_save_and_windows_clipboard_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(share, 'APP_DIR', Path(folder)):
            with patch.object(share.os, 'name', 'nt'), patch.object(share.subprocess, 'run') as run:
                share.save_url('https://sample-name.trycloudflare.com')
                run.assert_called_once_with(['clip.exe'], input=b'https://sample-name.trycloudflare.com', check=True)
                self.assertEqual((share.APP_DIR / 'public_url.txt').read_text().strip(), 'https://sample-name.trycloudflare.com')
                run.side_effect = OSError('clipboard unavailable')
                share.save_url('https://second.trycloudflare.com')
                self.assertIn('second', (share.APP_DIR / 'public_url.txt').read_text())

    def test_cloudflare_lifecycle_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(share, 'APP_DIR', Path(folder)), \
                patch.object(share.shutil, 'which', return_value='cloudflared'), \
                patch.object(share, 'start_streamlit') as start, \
                patch.object(share, 'wait_ready'), \
                patch.object(share.subprocess, 'Popen') as popen, \
                patch.object(share.time, 'sleep', side_effect=KeyboardInterrupt):
            start.return_value.poll.return_value = None
            tunnel = popen.return_value
            tunnel.poll.return_value = None
            tunnel.stdout = io.StringIO('https://test.trycloudflare.com\n')
            self.assertEqual(share.main(['--provider', 'cloudflare']), 0)
            start.return_value.terminate.assert_called_once()
            tunnel.terminate.assert_called_once()
            self.assertFalse((share.APP_DIR / 'public_url.txt').exists())

    def test_local_readiness_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(share, 'APP_DIR', Path(folder)), \
                patch.object(share, 'NGROK_TOKEN', ''), \
                patch.object(share, 'start_streamlit') as start, \
                patch.object(share, 'wait_ready', side_effect=RuntimeError('failed')):
            start.return_value.poll.return_value = None
            self.assertEqual(share.main([]), 1)
            start.return_value.terminate.assert_called_once()

    def test_ngrok_lifecycle_and_url(self):
        ngrok = Mock()
        ngrok.connect.return_value.public_url = 'https://sample.ngrok.app'
        with tempfile.TemporaryDirectory() as folder, patch.object(share, 'APP_DIR', Path(folder)), \
                patch.object(share, 'NGROK_TOKEN', 'test-token'), \
                patch.dict(sys.modules, {'pyngrok': Mock(ngrok=ngrok, conf=Mock())}), \
                patch.object(share, 'start_streamlit') as start, patch.object(share, 'wait_ready'), \
                patch.object(share.time, 'sleep', side_effect=KeyboardInterrupt), \
                patch.object(share, 'save_url') as save:
            start.return_value.poll.return_value = None
            self.assertEqual(share.main([]), 0)
            save.assert_called_once_with('https://sample.ngrok.app')
            ngrok.connect.assert_called_once_with(share.PORT, 'http', bind_tls=True)
            ngrok.disconnect.assert_called_once_with('https://sample.ngrok.app')
            ngrok.kill.assert_called_once()
            start.return_value.terminate.assert_called_once()

    def test_missing_cloudflared_no_server_started(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(share, 'APP_DIR', Path(folder)), \
                patch.object(share.shutil, 'which', return_value=None), patch.object(share, 'start_streamlit') as start:
            self.assertEqual(share.main(['--provider', 'cloudflare']), 1)
            start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
