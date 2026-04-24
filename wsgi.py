from smart_attendance import create_app


app = create_app()


if __name__ == "__main__":
    from waitress import serve

    serve(
        app,
        host=app.config["HOST"],
        port=app.config["PORT"],
        threads=8,
    )
