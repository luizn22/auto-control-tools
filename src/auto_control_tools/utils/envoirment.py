def is_jupyter_environment():
    try:
        # Attempt to import the IPython module
        import IPython
        # Check if IPython is running in interactive mode
        if IPython.get_ipython():
            return True
        else:
            return False
    except ImportError:
        # IPython is not available, so it's a normal Python environment
        return False
