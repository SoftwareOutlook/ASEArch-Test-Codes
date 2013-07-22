/* MPI halo exchange headers */

#ifdef USE_MPI
#define IN -1
#define OUT 1
#define UPDATE 1
#define NO_UPDATE 0
void post_recv(const struct grid_info_t *g);
void buffer_halo_transfers(const struct grid_info_t *g, int dir, int update);
void post_send(const struct grid_info_t *g);
void exchange_halos(const struct grid_info_t *g);
#endif
void compute_local_grid_ranges( struct grid_info_t * g );
