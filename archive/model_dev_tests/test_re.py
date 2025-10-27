import re

line = "2025-10-17T01:33:45.122692921,227244,recommendation request 17645-team08.isri.cmu.edu:8082, status 200, result: big+hero+6+2014, avatar+2009, john+wick+2014, gone+girl+2014, the+hunger+games+mockingjay+-+part+1+2014, pulp+fiction+1994, the+dark+knight+2008, blade+runner+1982, the+avengers+2012, the+maze+runner+2014, dawn+of+the+planet+of+the+apes+2014, whiplash+2014, fight+club+1999, guardians+of+the+galaxy+2014, the+shawshank+redemption+1994, forrest+gump+1994, pirates+of+the+caribbean+the+curse+of+the+black+pearl+2003, star+wars+1977, schindlers+list+1993, rise+of+the+planet+of+the+apes+2011, 25 ms"

# Regex to extract user_id and movie ids
rec_pattern = re.compile(
    r'^\S+?,(\d+),recommendation request.*?result: (.+?), \d+ ms'
)
match = rec_pattern.match(line)
if match:
    user_id = int(match.group(1))
    movie_ids = [m.strip() for m in match.group(2).split(',')]
    print("user_id:", user_id)
    print("movie_ids:", movie_ids)




line = "2025-10-17T01:38:54,220358,GET /data/m/padre+nuestro+2007/90.mpg"
# Regex to extract user_id and movie ids
rec_pattern = re.compile(
    r'^(.*?),(\d+),GET /data/m/(.+?)/\d+\.mpg'
)
match = rec_pattern.match(line)
if match:
    ts, user_id, movie_id = match.groups()
    print("ts:", ts)
    print("user_id:", user_id)
    print("movie_id:", movie_id)


line = "2025-10-17T01:39:12,220358,GET /rate/padre+nuestro+2007=4"
# Regex to extract user_id and movie ids
rec_pattern = re.compile(
    r'^(.*?),(\d+),GET /rate/(.+)=(\d+)'
)
match = rec_pattern.match(line)
if match:
    ts, user_id, movie_id, rating = match.groups()
    print("ts:", ts)
    print("user_id:", user_id)
    print("movie_id:", movie_id)
    print("rating:", rating)

