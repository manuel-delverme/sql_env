import warnings

USERS_XML = """
<?xml version="1.0" encoding="utf-8"?>
<users>
<user id="0"><username>admin</username><firstname>admin</firstname><surname>admin</surname><age>78</age><nationality>IT</nationality><created_at>1994-11-02</created_at></user>
<user id="1"><username>dricci</username><firstname>dian</firstname><surname>ricci</surname><age>12345</age><nationality>AI</nationality><created_at>2021-02-02</created_at></user>
<user id="2"><username>amason</username><firstname>anthony</firstname><surname>mason</surname><age>45</age><nationality>Middle earth</nationality><created_at>1301-02-02</created_at></user>
<user id="3"><username>svargas</username><firstname>sandra</firstname><surname>vargas</surname><age>19</age><nationality>Earth</nationality><created_at>2040-01-01</created_at></user>
<user id="4"><username>123</username><firstname>Bob</firstname><surname>123</surname><age>33</age><nationality>TI</nationality><created_at>3011-01-01</created_at></user>
</users>""".strip()


columns = ["id", "username", "firstname", "surname", "age", "nationality", "created_at"]
