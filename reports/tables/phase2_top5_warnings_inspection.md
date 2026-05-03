# Phase 2 — Top 5 films by `parse_warning_count`: raw XML vs parsed output

This report dumps source XML alongside the parser's output for the five films with the highest warning counts in the corpus. Its purpose is to confirm that the parser's recovery rules produced sensible output in the worst empirical cases.

Each entry shows: the film and its top-line counts, the first five warnings emitted during parsing, and for each of the first two scenes named in those warnings, the raw XML alongside the parsed `Scene` object's output.


## 1. 'Julieta_2016' (`tt4326444`) — 420 warnings

- `n_scenes`: 122
- `n_dialogue_lines`: 884
- `n_unique_characters`: 33
- `total_dialogue_chars`: 86,124
- `mean_dialogue_line_length`: 97.4


### First 5 warnings

- scene 2: rejected implausible character name '2016. SPRING'
- scene 2: <dialogue> with no attributable speaker
- scene 2: <dialogue> with no attributable speaker
- scene 2: <dialogue> with no attributable speaker
- scene 2: <dialogue> with no attributable speaker

### Scene 2 — raw XML

```xml
<scene>
    <stage_direction>1. MADRID. JULIETA'S HOUSE 2. INT. IN THE MORNING.</stage_direction>
    <character>2016. SPRING</character>
    <dialogue>A red fabric fills the screen. Over it appear the opening credits. At first it gives a sensation of stillness, but with the insistence of the shot we discover that the fabric is moving, a slight, rhythmic movement. We discover that the fabric is the front of a dress and that Julieta's heart is beating inside it.</dialogue>
    <dialogue>Julieta, an attractive woman of 55, independent and full of determination, a mixture of timidity and daring, fragility and courage. Blond. She is sitting next to a bookcase, surrounded by cardboard boxes, the kind used for moving house. She picks up a sculpture of naked, seated man, with the color and texture of terracotta (some 8 inches high), and wraps it carefully in bubble wrap. She places it in one of the cardboard boxes that contains art books, a framed photo with Lorenzo, a book of photos by Nan Goldin, items that she doesn't want to get rid of.</dialogue>
    <dialogue>In front of the shelves on which books are piled up in various groups, Julieta tries to decide which she'll put into boxes and which she'll leave. A great number of the books have to do with Classical Greece, Mythology, Art, etc. Juliet also puts in the box a novel by Lorenzo Gentile, "Adiós, Volcán", on the cover of which there is a sculpture by Ava, as we will discover in due time, also a naked, seated man.</dialogue>
    <dialogue>The walls of the apartment are painted white. A sober space with little furniture. Bright and comfortable. Few decorative elements and the few that we see have to do with Lorenzo. In one corner there is a large writing desk and, hanging in the angle formed by the wall, there are three posters. The rest of the walls are bare. A self-portrait of Lucien Freud announces a portrait exhibition in London, another poster by the hyper-realist painter Antonio López shows the deserted Gran Vía
... (truncated)
```

### Scene 2 — parsed output

```python
stage_direction:   "1. MADRID. JULIETA'S HOUSE 2. INT. IN THE MORNING."
scene_description: ''
n_dialogue_units:  6
First 8 dialogue_units:
  ('', 'A red fabric fills the screen. Over it appear the opening credits. At first it g')
  ('', 'Julieta, an attractive woman of 55, independent and full of determination, a mix')
  ('', 'In front of the shelves on which books are piled up in various groups, Julieta t')
  ('', 'The walls of the apartment are painted white. A sober space with little furnitur')
  ('', 'Julieta goes over to the writing desk and opens a side drawer. She rummages in t')
  ('', 'The entry phone rings. Julieta goes into the kitchen, answers and then opens the')
```

### Scene 3 — raw XML

```xml
<scene>
    <stage_direction>2. MADRID. JULIETA'S HOUSE 2. INT. IN THE MORNING.</stage_direction>
    <character>2016. SPRING.</character>
    <dialogue>Lorenzo Gentile in person comes into the apartment. Julieta had previously left the door open. He is aged between 55 and 60, an attractive man, warm-hearted and sincere. A mature, comfortable seducer. Argentinean, with a soft accent, he enjoys and is touched by the sight of Julieta, confused by the preparations for the journey. With a few books in her hand:</dialogue>
    <character>JULIETA</character>
    <dialogue>I'm in a real mess. I don't know which books to take.</dialogue>
    <character>LORENZO</character>
    <dialogue>Take the essential ones. If you miss any you can buy them on the Internet.</dialogue>
    <character>JULIETA</character>
    <dialogue>I don't like buying books I already have. It makes me feel old.</dialogue>
    <character>LORENZO</character>
    <dialogue>(Smiling) Right now you look like a little girl.</dialogue>
    <scene_description>Julieta makes a nervous face.</scene_description>
    <character>LORENZO</character>
    <dialogue>Apart from the books, do you have a lot to do?</dialogue>
    <character>JULIETA</character>
    <dialogue>I still have to do some shopping.</dialogue>
    <character>LORENZO</character>
    <dialogue>Julieta, we're not going to the desert. You'll be able to come to Madrid when you like… or go to Braga, which is closer.</dialogue>
    <character>JULIETA</character>
    <dialogue>(Suddenly, serious) I'd like not to come back to Madrid, if I can avoid it.</dialogue>
    <scene_description>Lorenzo embraces her tenderly. Julieta relaxes in his arms. They kiss.</scene_description>
    <character>LORENZO</character>
    <dialogue>(Murmuring to her) Thank you.</dialogue>
    <character>JULIETA</character>
    <dialogue>For what?</dialogue>
    <character>LORENZO</character>
    <dialogue>For not letting me grow old on my own.</dialogue>
    <character>JULIETA</charac
... (truncated)
```

### Scene 3 — parsed output

```python
stage_direction:   "2. MADRID. JULIETA'S HOUSE 2. INT. IN THE MORNING."
scene_description: 'Julieta makes a nervous face. Lorenzo embraces her tenderly. Julieta relaxes in his arms. They kiss.'
n_dialogue_units:  13
First 8 dialogue_units:
  ('', 'Lorenzo Gentile in person comes into the apartment. Julieta had previously left ')
  ('JULIETA', "I'm in a real mess. I don't know which books to take.")
  ('LORENZO', 'Take the essential ones. If you miss any you can buy them on the Internet.')
  ('JULIETA', "I don't like buying books I already have. It makes me feel old.")
  ('LORENZO', '(Smiling) Right now you look like a little girl.')
  ('LORENZO', 'Apart from the books, do you have a lot to do?')
  ('JULIETA', 'I still have to do some shopping.')
  ('LORENZO', "Julieta, we're not going to the desert. You'll be able to come to Madrid when yo")
```

## 2. 'Toy Story 4_2019' (`tt1979376`) — 235 warnings

- `n_scenes`: 55
- `n_dialogue_lines`: 1,196
- `n_unique_characters`: 69
- `total_dialogue_chars`: 42,116
- `mean_dialogue_line_length`: 35.2


### First 5 warnings

- scene 1: rejected implausible character name '©2019 DISNEY\x80�PIXAR'
- scene 1: <dialogue> with no attributable speaker
- scene 1: rejected implausible character name '©2019 DISNEY\x80�PIXAR'
- scene 1: <dialogue> with no attributable speaker
- scene 1: rejected implausible character name '©2019 DISNEY\x80�PIXAR'

### Scene 1 — raw XML

```xml
<scene>
    <scene_description>Toy Story 4 Written by Andrew Stanton Stephany Folsom ON BLACK Lightning flash! Torrential rain. Dark skies. CHYRON: Nine Years Ago The tone is ominous and quiet. Too quiet. We PULL BACK to reveal...ANDY'S ROOM JESSIE and BULLSEYE stare out the window. Worried.</scene_description>
    <character>JESSIE</character>
    <dialogue>Whoa! It's raining cats and dogs out there! I hope they make it back alright...</dialogue>
    <scene_description>FOOTSTEPS. Coming fast. The few toys littered about the room race back to their places.</scene_description>
    <character>HAMM</character>
    <dialogue>Heads up! Andy's coming!</dialogue>
    <scene_description>ANDY (8) bursts in. Slightly wet, but triumphant. Dumps an arm-full of toys on his bed: WOODY, BUZZ, REX, SLINKY, THE POTATO HEADS, THE ALIENS. Equally wet. Stained with grass and dirt.</scene_description>
    <character>ANDY'S MOM (O.S.)</character>
    <dialogue>Andy! Time for dinner.</dialogue>
    <character>ANDY</character>
    <dialogue>Yes! I'm starving!</dialogue>
    <scene_description>Andy runs out, leaving the door ajar. We LISTEN TO THE FOOTSTEPS descend...fade away...and... THUNDER -- THE TOYS JUMP TO LIFE IN A COMPLETE PANIC!! Woody's already at the windowsill, searching. Buzz is close behind.</scene_description>
    <character>BUZZ</character>
    <dialogue>Do you see him?</dialogue>
    <character>WOODY</character>
    <dialogue>No.</dialogue>
    <character>SLINKY DOG</character>
    <dialogue>Well, he's done for.</dialogue>
    <character>©2019 DISNEY�PIXAR</character>
    <dialogue>REX</dialogue>
    <scene_description>He'll be lost! Forever!</scene_description>
    <character>WOODY</character>
    <dialogue>Jessie. Buzz. Slink. Molly's room.</dialogue>
    <scene_description>Woody is already on the move.</scene_description>
    <character>WOODY</character>
    <dialogue>The rest of you stay put.</dialogue>
    <scene_description>ON UPSTAIRS HALLWAY Woody checks the coast
... (truncated)
```

### Scene 1 — parsed output

```python
stage_direction:   ''
scene_description: 'Toy Story 4 Written by Andrew Stanton Stephany Folsom ON BLACK Lightning flash! Torrential rain. Dark skies. CHYRON: Nine Years Ago The tone is ominous and quiet. Too quiet. We PULL BACK to reveal...A'
n_dialogue_units:  25
First 8 dialogue_units:
  ('JESSIE', "Whoa! It's raining cats and dogs out there! I hope they make it back alright...")
  ('HAMM', "Heads up! Andy's coming!")
  ("ANDY'S MOM", 'Andy! Time for dinner.')
  ('ANDY', "Yes! I'm starving!")
  ('BUZZ', 'Do you see him?')
  ('WOODY', 'No.')
  ('SLINKY DOG', "Well, he's done for.")
  ('', 'REX')
```

### Scene 4 — raw XML

```xml
<scene>
    <stage_direction>..Mom grab Bo Peep.</stage_direction>
    <character>VISITOR FATHER</character>
    <dialogue>Oh, it's beautiful...</dialogue>
    <character>ANDY'S MOM</character>
    <dialogue>...I'm so glad to see this old lamp go to a good home. We've had it since Molly was a baby.</dialogue>
    <scene_description>To Woody's shock; Mom places Bo, her sheep, and lamp into a CARDBOARD BOX.</scene_description>
    <character>VISITOR FATHER</character>
    <dialogue>Molly, are you sure it's alright?</dialogue>
    <character>©2019 DISNEY�PIXAR</character>
    <dialogue>MOLLY</dialogue>
    <scene_description>Yeah, I don't want it anymore. Woody watches, helpless, as Mom hands off the box to the Visitor Father. They head back downstairs... INSIDE The toys jump up! Throw open the window! Pull up Slinky Dog and...nobody else?</scene_description>
    <character>BUZZ</character>
    <dialogue>Where's Woody?</dialogue>
    <scene_description>OUTSIDE The Visitor Father walks to the trunk of his car. Sets the box down to search his pockets. Jogs back to the house and knocks. Mom answers...</scene_description>
    <character>VISITOR FATHER</character>
    <dialogue>Yeah hi, I think I left my keys in here...</dialogue>
    <scene_description>In that unattended moment, BO'S BOX IS SUDDENLY DRAGGED... UNDER THE CAR LIGHTNING FLASHES to reveal Woody opening the box. Bo is comforting her sheep in the darkness. WOODY-- !</scene_description>
    <character>WOODY</character>
    <dialogue>Quick! We'll sneak in the hedges before he's back--</dialogue>
    <character>BO</character>
    <dialogue>Woody, it's okay...</dialogue>
    <character>WOODY</character>
    <dialogue>Wha--? No! No, no. You can't go. What's best for Andy is that you--</dialogue>
    <character>BO</character>
    <dialogue>Woody. I'm not Andy's toy.</dialogue>
    <character>WOODY</character>
    <dialogue>Wha-- What?</dialogue>
    <scene_description>Woody goes still. Looks at Bo. She's right.</scen
... (truncated)
```

### Scene 4 — parsed output

```python
stage_direction:   '..Mom grab Bo Peep.'
scene_description: "To Woody's shock; Mom places Bo, her sheep, and lamp into a CARDBOARD BOX. Yeah, I don't want it anymore. Woody watches, helpless, as Mom hands off the box to the Visitor Father. They head back downst"
n_dialogue_units:  22
First 8 dialogue_units:
  ('VISITOR FATHER', "Oh, it's beautiful...")
  ("ANDY'S MOM", "...I'm so glad to see this old lamp go to a good home. We've had it since Molly ")
  ('VISITOR FATHER', "Molly, are you sure it's alright?")
  ('', 'MOLLY')
  ('BUZZ', "Where's Woody?")
  ('VISITOR FATHER', 'Yeah hi, I think I left my keys in here...')
  ('WOODY', "Quick! We'll sneak in the hedges before he's back--")
  ('BO', "Woody, it's okay...")
```

## 3. 'Iron Man_2008' (`tt0371746`) — 118 warnings

- `n_scenes`: 148
- `n_dialogue_lines`: 906
- `n_unique_characters`: 83
- `total_dialogue_chars`: 53,163
- `mean_dialogue_line_length`: 58.7


### First 5 warnings

- scene 3: character 'RHODEY' followed by another <character>
- scene 3: rejected implausible character name '© 2007 MARVEL STUDIOS, INC.'
- scene 5: rejected implausible character name '© 2007 MARVEL STUDIOS, INC.'
- scene 5: rejected implausible character name '© 2007 MARVEL STUDIOS, INC.'
- scene 7: rejected implausible character name '© 2007 MARVEL STUDIOS, INC.'

### Scene 3 — raw XML

```xml
<scene>
    <stage_direction>INT. HUMMER - CONTINUOUS</stage_direction>
    <scene_description>Three Airmen, kids with battle-worn faces. Crammed in there with them is a Man in an expensive suit, who looks tele- ported from Beverly Hills. He is, of course, genius inventor and billionaire, TONY STARK. In his hand is a drink tumbler of vodka.</scene_description>
    <character>TONY</character>
    <dialogue>Oh, I get it. You guys aren't allowed to talk. Is that it? Are you not allowed to talk?</dialogue>
    <scene_description>One Airman grins, fidgeting with his orange NY Mets watch.</scene_description>
    <character>JIMMY</character>
    <dialogue>No. We're allowed to talk.</dialogue>
    <character>TONY</character>
    <dialogue>Oh. I see. So it's personal.</dialogue>
    <character>RAMIREZ</character>
    <dialogue>I think they're intimidated.</dialogue>
    <character>TONY</character>
    <dialogue>Good God, you're a woman.</dialogue>
    <scene_description>The others try to compress laughs.</scene_description>
    <character>TONY</character>
    <dialogue>I, honestly, I couldn't have called that.</dialogue>
    <parenthetical>(after silence)</parenthetical>
    <dialogue>I would apologize, but isn't that what we're going for here? I saw you as a soldier first.</dialogue>
    <character>JIMMY</character>
    <dialogue>I have a question, sir.</dialogue>
    <character>TONY</character>
    <dialogue>Please.</dialogue>
    <scene_description>NO DUPLICATION WITHOUT MARVEL'S WRITTEN CONSENT. CONTINUED:</scene_description>
    <character>JIMMY</character>
    <dialogue>Is it true you're twelve for twelve with last years Maxim cover girls?</dialogue>
    <character>TONY</character>
    <dialogue>Excellent question. Yes and no. March and I had a schedule conflict but, thankfully, the Christmas cover was twins. Anyone else? You, with the hand up.</dialogue>
    <character>PRATT</character>
    <dialogue>It's a little embarrassing.</dialogue>
    <character>TONY</characte
... (truncated)
```

### Scene 3 — parsed output

```python
stage_direction:   'INT. HUMMER - CONTINUOUS'
scene_description: 'Three Airmen, kids with battle-worn faces. Crammed in there with them is a Man in an expensive suit, who looks tele- ported from Beverly Hills. He is, of course, genius inventor and billionaire, TONY '
n_dialogue_units:  18
First 8 dialogue_units:
  ('TONY', "Oh, I get it. You guys aren't allowed to talk. Is that it? Are you not allowed t")
  ('JIMMY', "No. We're allowed to talk.")
  ('TONY', "Oh. I see. So it's personal.")
  ('RAMIREZ', "I think they're intimidated.")
  ('TONY', "Good God, you're a woman.")
  ('TONY', "I, honestly, I couldn't have called that.")
  ('TONY', "I would apologize, but isn't that what we're going for here? I saw you as a sold")
  ('JIMMY', 'I have a question, sir.')
```

### Scene 5 — raw XML

```xml
<scene>
    <stage_direction>INT. INSURGENT CAVE - AFGHANISTAN - DAY</stage_direction>
    <scene_description>Tony snaps awake. He's tied to a chair, bloody rags covering his chest. Two Insurgents flank a DV camera. Behind Tony -- A line of armed hooded men and a banner showing ten interlocked rings. The Leader, a huge Choori knife in one hand, reads rhetoric (in Dari) for the camera. PUSH IN ON - THE DV CAMERA VIEWFINDER: until the image of a desperate Tony breaks up into pixel chaos. CUT TO: CREDITS OVER A FULL SCREEN FILM REEL: The attack on Pearl Harbor. FDR gives an impassioned speech.</scene_description>
    <character>NARRATOR (RHODEY'S VOICE)</character>
    <dialogue>December 7, 1941: the day the world changed forever.</dialogue>
    <scene_description>NO DUPLICATION WITHOUT MARVEL'S WRITTEN CONSENT. CONTINUED:</scene_description>
    <character>NARRATOR (RHODEY'S VOICE)</character>
    <dialogue>President Roosevelt declares the United States will build fifty thousand planes to fight the armies of Hirohito and Hitler--</dialogue>
    <scene_description>S.S. Officers goose-step through Paris.</scene_description>
    <character>NARRATOR</character>
    <dialogue>Although no such capacity to build existed...</dialogue>
    <scene_description>1940s L.A., an unassuming hangar reads: "STARK INDUSTRIES."</scene_description>
    <character>NARRATOR</character>
    <dialogue>Howard Stark, founder of the fledgling Stark Industries, answers his call to duty --</dialogue>
    <scene_description>A Young Howard Stark shakes FDR's hand.</scene_description>
    <character>NARRATOR</character>
    <dialogue>And builds not fifty, but a hundred thousand planes.</dialogue>
    <scene_description>An airfield covered in B-29s. Stark bombers in flight, strewing bombs and paratroopers across the sky.</scene_description>
    <character>NARRATOR</character>
    <dialogue>Later, Stark's work on the Manhattan project makes the end of the war possible.</dialogue>
    <scene_description
... (truncated)
```

### Scene 5 — parsed output

```python
stage_direction:   'INT. INSURGENT CAVE - AFGHANISTAN - DAY'
scene_description: "Tony snaps awake. He's tied to a chair, bloody rags covering his chest. Two Insurgents flank a DV camera. Behind Tony -- A line of armed hooded men and a banner showing ten interlocked rings. The Lead"
n_dialogue_units:  15
First 8 dialogue_units:
  ('NARRATOR', 'December 7, 1941: the day the world changed forever.')
  ('NARRATOR', 'President Roosevelt declares the United States will build fifty thousand planes ')
  ('NARRATOR', 'Although no such capacity to build existed...')
  ('NARRATOR', 'Howard Stark, founder of the fledgling Stark Industries, answers his call to dut')
  ('NARRATOR', 'And builds not fifty, but a hundred thousand planes.')
  ('NARRATOR', "Later, Stark's work on the Manhattan project makes the end of the war possible.")
  ('NARRATOR', 'Stark Industries would go on to contribute to every major weapons system through')
  ('NARRATOR', "But Howard Stark's greatest achievement would come in 1973--")
```

## 4. 'Elvis_2022' (`tt3704428`) — 102 warnings

- `n_scenes`: 4
- `n_dialogue_lines`: 1,599
- `n_unique_characters`: 245
- `total_dialogue_chars`: 112,105
- `mean_dialogue_line_length`: 70.1


### First 5 warnings

- scene 4: rejected implausible character name 'EXT. SUN STUDIOS - DUSK (1954)'
- scene 4: <dialogue> with no attributable speaker
- scene 4: <dialogue> with no attributable speaker
- scene 4: rejected implausible character name 'INT. SUN STUDIOS - CONTROL ROOM - DUSK'
- scene 4: <dialogue> with no attributable speaker

### Scene 4 — raw XML

```xml
<scene>
    <stage_direction>INT. AMBULANCE - CONTINUOUS ACTION</stage_direction>
    <scene_description>LOOKING DOWN ON: Colonel. Eyes closed. Dead-looking. MEDICS attempt to revive him.</scene_description>
    <character>CONTINUED:</character>
    <dialogue>OLD COLONEL (V.O.)</dialogue>
    <scene_description>There are some who'd make me out to be the villain of this here story...</scene_description>
    <character>EXT. INTERNATIONAL HOTEL - DUSK</character>
    <dialogue>The ambulance races past the International Hotel. The CAMERA WHIP PANS TO its towering sign: "THE STAR TREK EXPERIENCE: BOLDLY GOING WHERE NO MAN HAS GONE BEFORE." The sign SPINS ON ITS AXIS as we JOURNEY BACK TO the International of the 1970s. The sign now heralding: "ELVIS!"</dialogue>
    <character>EXT. INTERNATIONAL HOTEL - PORTE COCHERE - NIGHT (1974)</character>
    <dialogue>We STREAM ALONG WITH the ELVIS FANS pouring out of limos and THROUGH the hotel's glass doors...</dialogue>
    <character>OLD COLONEL (V.O.)</character>
    <dialogue>Who'd say I exploited the boy and stole all his money...</dialogue>
    <scene_description>A CAMERA CREW shoots 16MM FOOTAGE of CROWDS being ushered through the lobby. QUICK CUTS of every possible permutation of merchandise being snatched up by the adoring crowds.</scene_description>
    <character>OLD COLONEL (V.O.)</character>
    <dialogue>Trapped him in Vegas and enabled his drug addiction...</dialogue>
    <character>INT. INTERNATIONAL HOTEL - SERVICE CORRIDOR - NIGHT</character>
    <dialogue>SPLASH! We're UNDERWATER. A DROWNED MAN, pale face obscured by long, black tendrils of hair...</dialogue>
    <character>INT. INTERNATIONAL HOTEL - SHOWROOM - NIGHT</character>
    <dialogue>Onstage, a WARM-UP COMEDIAN cracks cheesy gags.</dialogue>
    <character>INT. INTERNATIONAL HOTEL - CASINO - NIGHT</character>
    <dialogue>By a roped-off craps table, a gold trolley stacked high with chips. A small crowd looks on as a gruff-looking security guard, RED W
... (truncated)
```

### Scene 4 — parsed output

```python
stage_direction:   'INT. AMBULANCE - CONTINUOUS ACTION'
scene_description: "LOOKING DOWN ON: Colonel. Eyes closed. Dead-looking. MEDICS attempt to revive him. There are some who'd make me out to be the villain of this here story... A CAMERA CREW shoots 16MM FOOTAGE of CROWDS "
n_dialogue_units:  1591
First 8 dialogue_units:
  ('CONTINUED:', 'OLD COLONEL (V.O.)')
  ('EXT. INTERNATIONAL HOTEL - DUSK', 'The ambulance races past the International Hotel. The CAMERA WHIP PANS TO its to')
  ('EXT. INTERNATIONAL HOTEL - PORTE COCHERE - NIGHT', "We STREAM ALONG WITH the ELVIS FANS pouring out of limos and THROUGH the hotel's")
  ('OLD COLONEL', "Who'd say I exploited the boy and stole all his money...")
  ('OLD COLONEL', 'Trapped him in Vegas and enabled his drug addiction...')
  ('INT. INTERNATIONAL HOTEL - SERVICE CORRIDOR - NIGHT', "SPLASH! We're UNDERWATER. A DROWNED MAN, pale face obscured by long, black tendr")
  ('INT. INTERNATIONAL HOTEL - SHOWROOM - NIGHT', 'Onstage, a WARM-UP COMEDIAN cracks cheesy gags.')
  ('INT. INTERNATIONAL HOTEL - CASINO - NIGHT', 'By a roped-off craps table, a gold trolley stacked high with chips. A small crow')
```

## 5. 'Cat People_1942' (`tt0034587`) — 62 warnings

- `n_scenes`: 128
- `n_dialogue_lines`: 540
- `n_unique_characters`: 33
- `total_dialogue_chars`: 33,351
- `mean_dialogue_line_length`: 61.8


### First 5 warnings

- scene 1: character 'THE CAT PEOPLE' had no following <dialogue>
- scene 1: character 'EVEN AS FOG CONTINUES TO LIE IN THE VALLEYS, SO DOES ANCIENT' followed by another <character>
- scene 1: character 'SIN CLING TO THE LOW PLACES, THE DEPRESSIONS IN THE WORLD' followed by another <character>
- scene 1: character 'CONSCIOUSNESS.' followed by another <character>
- scene 1: scene ended with character 'SIGMUND FREUD' and no following <dialogue>

### Scene 1 — raw XML

```xml
<scene>
    <scene_description>brought to you by The Val Lewton Screenplay Collection</scene_description>
    <character>THE CAT PEOPLE</character>
    <scene_description>Original Screen Play  by DeWitt Bodeen  The RKO trademark FADES OFF, leaving a black screen, in the center of which are two slits of pale light. These move closer until we see that they are a pair of cat's eyes. Over these mysteriously blinking lights the title is SUPERIMPOSED.</scene_description>
    <character>DISSOLVE</character>
    <dialogue>A misty OUT-OF-FOCUS SHOT of a black panther pacing behind cage bars. Over this come the production credits. A pale fog rises over the shot of the black panther, and over it Is SUPERIMPOSED the following quotation:</dialogue>
    <character>EVEN AS FOG CONTINUES TO LIE IN THE VALLEYS, SO DOES ANCIENT</character>
    <character>SIN CLING TO THE LOW PLACES, THE DEPRESSIONS IN THE WORLD</character>
    <character>CONSCIOUSNESS.</character>
    <character>SIGMUND FREUD</character>
  </scene>
  
```

### Scene 1 — parsed output

```python
stage_direction:   ''
scene_description: 'brought to you by The Val Lewton Screenplay Collection Original Screen Play  by DeWitt Bodeen  The RKO trademark FADES OFF, leaving a black screen, in the center of which are two slits of pale light. '
n_dialogue_units:  6
First 8 dialogue_units:
  ('THE CAT PEOPLE', '')
  ('DISSOLVE', 'A misty OUT-OF-FOCUS SHOT of a black panther pacing behind cage bars. Over this ')
  ('EVEN AS FOG CONTINUES TO LIE IN THE VALLEYS, SO DOES ANCIENT', '')
  ('SIN CLING TO THE LOW PLACES, THE DEPRESSIONS IN THE WORLD', '')
  ('CONSCIOUSNESS.', '')
  ('SIGMUND FREUD', '')
```

### Scene 2 — raw XML

```xml
<scene>
    <stage_direction>EXT. ZOO PROMENADE - PARK - AFTERNOON</stage_direction>
    <scene_description>As the last word of the quotation FADES from the screen, the fog clears, the caged leopard comes into full focus, and we see that it is an actual leopard behind actual bars. Over the scene is the wheezy music of the Triumphal March from "Aida," as played on a hand, organ. This is playing in the distance, and we do not see the organ-grinder until later. The CAMERA DRAWS BACK to show a young artist sitting before the cage on a campstool with a drawing portfolio in her hand. She is presumably sketching the panther, although her drawing is not shown, and we do not see the features of the girl's face. The girl picks up the drawing and holds it off, weighing its values. It evidently does not meet with her approval, for she wads the drawing Into a ball and turns to look for a place to throw the waste paper. We see her face. It is heart-shaped, demure, even a little naive. She is small, young,and very beautiful. In one hand the wad of waste paper is poised, ready to throw into a container. INSERT WASTEPAPER BASKET as Irena sees it. It is a rather fancy container in the shape of a tree trunk.</scene_description>
    <character>OLIVER</character>
    <dialogue>Yes. (then continuing in the same tone)</dialogue>
    <dialogue>"Sometimes whoever seeks abroad may find Thee sitting careless on a granary floor, Thy hair soft lifted... (ponders, as if trying to remember) Thy hair soft lifted..."</dialogue>
    <character>ALICE</character>
    <parenthetical>(snapping her compact shut)</parenthetical>
    <dialogue>Reminds me. I have a date with the hairdresser.</dialogue>
    <character>OLIVER</character>
    <dialogue>What a way to spend a Saturday afternoon!</dialogue>
    <character>ALICE</character>
    <dialogue>The business girl's holiday</dialogue>
    <character>OLIVER</character>
    <dialogue>You should've minded your mother and eaten more bread crusts. You'd have cu
... (truncated)
```

### Scene 2 — parsed output

```python
stage_direction:   'EXT. ZOO PROMENADE - PARK - AFTERNOON'
scene_description: 'As the last word of the quotation FADES from the screen, the fog clears, the caged leopard comes into full focus, and we see that it is an actual leopard behind actual bars. Over the scene is the whee'
n_dialogue_units:  14
First 8 dialogue_units:
  ('OLIVER', 'Yes. (then continuing in the same tone)')
  ('OLIVER', '"Sometimes whoever seeks abroad may find Thee sitting careless on a granary floo')
  ('ALICE', 'Reminds me. I have a date with the hairdresser.')
  ('OLIVER', 'What a way to spend a Saturday afternoon!')
  ('ALICE', "The business girl's holiday")
  ('OLIVER', "You should've minded your mother and eaten more bread crusts. You'd have curly h")
  ('ALICE', 'Thank you for lunch. See you at the office Monday.')
  ('IRENA', 'Thank you.')
```