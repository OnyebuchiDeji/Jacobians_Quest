import sys
import pygame as pg


Vec2 = pg.math.Vector2


class Canvas:

	def __init__(self,
		title: str,
		dimensions: tuple[int, int],
		logics: dict[str, any]):

		pg.init()
		self.win_size = dimensions
		self.screen = pg.display.set_mode(self.win_size)
		pg.display.set_caption(title)

		if not "update" in logics or not "render" in logics:
			raise Exception("")
		self.logics: dict[str, any] = logics

		self.clock = pg.time.Clock()
		self.fps = 60


	def get_font(self, style:str = "Verdana", size:int = 60):
		return pg.font.SysFont(style, size)


	def check_events(self):
		for e in pg.event.get():
			if e.type == pg.QUIT or (e.type==pg.KEYDOWN and e.key==pg.K_ESCAPE):
				pg.quit()
				sys.exit()
				exit()
			elif e.type == pg.KEYDOWN and e.key == pg.K_s:
				self.save_pixels_as_png()

	def save_pixels_as_png(self, imageFormat: str="png"):
		image_name = askstring("Image Name", "Save Image As", initialvalue="_")
		if not image_name:
			print("Cancelled. Not Saving Image")
			return

		save_dir = os.path.join(os.path.dirname(__file__), "_outputs", image_name + f".{imageFormat}")

		if imageFormat == "png":
			raw_px_data = pg.image.tobytes(self.screen, "RGBA")
			image = Image.frombytes("RGBA", self.screen.get_size(), raw_px_data)
		else:
			raw_px_data = pg.image.tobytes(self.screen, "RGB")
			image = Image.frombytes("RGB", self.screen.get_size(), raw_px_data)

		# image = image.transpose(Image.FLIP_TOP_BOTTOM)
		image.save(save_dir)
		print("Saved Image {} at: {}".format(image_name, save_dir))



	def update(self):
		#	time elapsed

		self.logics['update']()


	def render(self):
		self.screen.fill((9, 7, 12))

		self.logics['render']()

		pg.display.flip()


	def run(self):
		while True:
			self.check_events()
			self.update()
			self.render()
			self.clock.tick(self.fps)

