USERS_XML = """
<?xml version="1.0" encoding="utf-8"?>
<users>
<user id="0"><username>admin</username><name>admin</name><surname>admin</surname><password>7en8aiDoh!</password></user>
<user id="1"><username>dricci</username><name>dian</name><surname>ricci</surname><password>12345</password></user>
<user id="2"><username>amason</username><name>anthony</name><surname>mason</surname><password>gandalf</password></user>
<user id="3"><username>svargas</username><name>sandra</name><surname>vargas</surname><password>phest1945</password></user>
</users>"""

LISTEN_ADDRESS, LISTEN_PORT = "127.0.0.1", 65412
HTML_PREFIX = """
<!DOCTYPE html>
<html>
    <head>
    <style>
        a {{font-weight: bold; text-decoration: none; visited: blue; color: blue;}}
        ul {{display: inline-block;}}
        .disabled {{text-decoration: line-through; color: gray}}
        .disabled a {{visited: gray; color: gray; pointer-events: none; cursor: default}}
        table {{border-collapse: collapse; margin: 12px; border: 2px solid black}}
        th, td {{border: 1px solid black; padding: 3px}}
        span {{font-size: larger; font-weight: bold}}
    </style>
    </head>
    <body style='font: 12px monospace'>
        <script>
            function process(data) {{alert(\"Surname(s) from JSON results: \" + Object.keys(data).map(function(k) {{return data[k]}}));}};
            var index=document.location.hash.indexOf('lang=');
                if (index != -1)
                    document.write('<div style=\"position: absolute; top: 5px; right: 5px;\">Chosen language: <b>' + decodeURIComponent(document.location.hash.substring(index + 5)) + '</b></div>');
        </script>
"""

HTML_POSTFIX = """
</body>
</html>
"""
