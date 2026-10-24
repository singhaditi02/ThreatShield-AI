"""
app.py  ·  Section 7 of the notebook  ·  entry point
Run with:   python app.py
Options:    python app.py --share      (public gradio.live link)
            python app.py --port 8000
"""
import argparse

from threatshield.ui import demo


def main():
    parser = argparse.ArgumentParser(description="Launch ThreatShield AI")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860, help="local port (default 7860)")
    args = parser.parse_args()
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port,
                        share=args.share, debug=False, show_error=True)


if __name__ == "__main__":
    main()
