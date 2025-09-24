### This is a sample output of the content-based recommendation model based on old data as of Sept 22, 2025.

```plaintext

Building movie profiles...
Building user profiles...
Training completed in 0.17 seconds
Generating recommendations for existing user 142245
Recommendations generated in 0.0021 seconds

Recommendations for existing user 142245:
                                              movie_id                                             title  similarity
226                                     tommy+boy+1995                                         Tommy Boy    0.968651
967                                        friday+1995                                            Friday    0.968197
323                                        clerks+1994                                            Clerks    0.968010
302                          the+first+wives+club+1996                              The First Wives Club    0.967834
123                                  the+birdcage+1996                                      The Birdcage    0.967667
606         romy+and+micheles+high+school+reunion+1997            Romy and Michele's High School Reunion    0.967377
607                                       kingpin+1996                                           Kingpin    0.967360
29                                            uhf+1989                                               UHF    0.967181
982                               ruthless+people+1986                                   Ruthless People    0.967163
24           national+lampoons+christmas+vacation+1989             National Lampoon's Christmas Vacation    0.967058
977                        airplane+ii+the+sequel+1982                           Airplane II: The Sequel    0.966448
777                                vegas+vacation+1997                                    Vegas Vacation    0.966420
135  to+wong+foo_+thanks+for+everything+julie+newma...  To Wong Foo, Thanks for Everything! Julie Newmar    0.966194
510                                  best+in+show+2000                                      Best in Show    0.965832
525                   father+of+the+bride+part+ii+1995                       Father of the Bride Part II    0.965568
616                                   nine+months+1995                                       Nine Months    0.965517
223                   the+world+according+to+garp+1982                       The World According to Garp    0.965178
282                                     liar+liar+1997                                         Liar Liar    0.965103
344                               dumb+and+dumber+1994                                   Dumb and Dumber    0.965069
994                                    caddyshack+1980                                        Caddyshack    0.965026
No metadata for user 999999. Using generic profile.
Recommendations generated in 0.0006 seconds

Recommendations for new user (cold start):
                                  movie_id                              title  similarity
489                            avatar+2009                             Avatar    0.552003
928                   the+dark+knight+2008                    The Dark Knight    0.551556
468                         inception+2010                          Inception    0.546702
548                       i+am+legend+2007                        I Am Legend    0.533315
943             the+dark+knight+rises+2012              The Dark Knight Rises    0.532446
369  final+fantasy+the+spirits+within+2001  Final Fantasy: The Spirits Within    0.529615
231               the+matrix+reloaded+2003                The Matrix Reloaded    0.527268
989             young+sherlock+holmes+1985              Young Sherlock Holmes    0.523415
843                    the+negotiator+1998                     The Negotiator    0.523318
180                           aladdin+1992                            Aladdin    0.521999
381                           titanic+1997                            Titanic    0.520586
31                       the+avengers+2012                       The Avengers    0.517321
175    dawn+of+the+planet+of+the+apes+2014     Dawn of the Planet of the Apes    0.517213
454                  the+last+samurai+2003                   The Last Samurai    0.516198
774                      interstellar+2014                       Interstellar    0.515471
178              inglourious+basterds+2009               Inglourious Basterds    0.514871
904              x-men+the+last+stand+2006              X-Men: The Last Stand    0.513782
481                      total+recall+2012                       Total Recall    0.513027
332     the+girl+who+played+with+fire+2009      The Girl Who Played with Fire    0.512367
134                             shrek+2001                              Shrek    0.512137

Model Metrics:
Training Time: 0.17 seconds
Inference Time (existing user): 0.0021 seconds
Inference Time (cold start): 0.0006 seconds
Model Size: 0.47 MB